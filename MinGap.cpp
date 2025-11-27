#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include "vcl2/vectorclass.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "ApproxTools/Chebyshev.hpp"
#include <Spectra/SymEigsSolver.h>
#include <random>

#define PI 3.1415926535897932384626

using namespace Eigen;

//Custom struct for holding problem instances
struct ProblemInstance {
  //Problem size
  uint32_t n;

  //h and J parameters, total length n*(n+1)/2
  std::vector<float> params;

  //A polynomial describing the energy landscape
  Chebyshev<double> poly;

  //How many minima we save
  uint32_t num_minima;

  //The location and value of the minima
  std::vector<float> minima;

  void save(const std::string &filename) const;
  static ProblemInstance load(const std::string &filename);
};
class ProblemFile {
public:
  ProblemFile(const std::string& filename,
        uint32_t n,
        uint32_t polyDegree,
        uint32_t numMinima)
    : filename_(filename),
      n_(n),
      m_(polyDegree),
      k_(numMinima),
      count_(0)
  {
    std::ifstream in(filename_, std::ios::binary);

    if (!in.good()) {
      // File doesn't exist → create a new one with header
      writeHeader();
    } else {
      // File exists → validate header and determine instance count
      readHeader();
    }
  }

  size_t instanceCount() const { return count_; }

  // Save one problem to disk
  void saveInstance(const ProblemInstance& prob)
  {
    size_t numParams = n_ * (n_ + 1) / 2;

    if (prob.params.size() != numParams)
      throw std::runtime_error("params wrong size");

    if (prob.poly.coeffs.size() != m_ + 1){
      std::cout << prob.poly.coeffs.size() << " " << m_ << "\n\n";
      throw std::runtime_error("polynomial wrong size");
    }

    if (prob.minima.size() != 2 * k_)
      throw std::runtime_error("minima wrong size");

    std::ofstream out(filename_, std::ios::binary | std::ios::app);
    if (!out)
      throw std::runtime_error("Failed to open file for append.");

    out.write((char*)prob.params.data(), numParams * sizeof(float));
    out.write((char*)prob.poly.coeffs.data(), (m_ + 1) * sizeof(double));
    out.write((char*)prob.minima.data(), 2 * k_ * sizeof(float));

    count_++;
  }


  // Load a problem from disk
  void loadInstance(size_t index, ProblemInstance& prob) const
  {
    if (index >= count_)
      throw std::runtime_error("index out of range");

    size_t numParams = n_ * (n_ + 1) / 2;

    std::vector<double> polyCoeffs(m_ + 1);

    prob.n = n_;
    prob.num_minima = k_;
    prob.params.resize(numParams);
    prob.minima.resize(2 * k_);

    size_t instanceSize =
        sizeof(float) * numParams
      + sizeof(double) * (m_ + 1)
      + sizeof(float) * (2 * k_);

    size_t headerSize =
        sizeof(MAGIC)
      + sizeof(n_)
      + sizeof(m_)
      + sizeof(k_);

    size_t offset = headerSize + index * instanceSize;

    std::ifstream in(filename_, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for reading.");

    in.seekg(offset);
    in.read((char*)prob.params.data(), numParams * sizeof(float));
    in.read((char*)polyCoeffs.data(), (m_ + 1) * sizeof(double));
    in.read((char*)prob.minima.data(), 2 * k_ * sizeof(float));

    // Convert to Eigen vector and construct Chebyshev
    VectorXd vec = Eigen::Map<VectorXd>(polyCoeffs.data(), polyCoeffs.size());
    prob.poly = Chebyshev<double>(vec);
  }

private:
  std::string filename_;
  uint32_t n_, m_, k_;
  size_t count_;

  static constexpr uint32_t MAGIC = 0x504F4C59; // "POLY"

  //------------------------------------------------------
  void writeHeader()
  {
    std::ofstream out(filename_, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to create file.");

    out.write((char*)&MAGIC, sizeof(MAGIC));
    out.write((char*)&n_, sizeof(n_));
    out.write((char*)&m_, sizeof(m_));
    out.write((char*)&k_, sizeof(k_));

    count_ = 0;
  }

  //------------------------------------------------------
  void readHeader()
  {
    std::ifstream in(filename_, std::ios::binary);
    if (!in)
      throw std::runtime_error("Failed to open file.");

    uint32_t magic;
    in.read((char*)&magic, sizeof(magic));

    if (magic != MAGIC)
      throw std::runtime_error("Invalid file: bad magic number.");

    in.read((char*)&n_, sizeof(n_));
    in.read((char*)&m_, sizeof(m_));
    in.read((char*)&k_, sizeof(k_));

    // Determine instance count
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();

    size_t headerSize =
        sizeof(MAGIC)
      + sizeof(n_)
      + sizeof(m_)
      + sizeof(k_);

    size_t numParams = n_ * (n_ + 1) / 2;

    size_t instanceSize =
        sizeof(float) * numParams
      + sizeof(double) * (m_ + 1)
      + sizeof(float) * (2 * k_);

    if ((size - headerSize) % instanceSize != 0)
      throw std::runtime_error("File corrupted or incompatible.");

    count_ = (size - headerSize) / instanceSize;
  }
};

//Get least significant bit
unsigned int LSB(int n){
  return n & (-n);
}

//Convert value to grey code
unsigned int grey(unsigned int n){
  return n ^ (n >> 1);
}

//Get position of the only set bit
//Taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
unsigned int log2(unsigned int v){
  static const int MultiplyDeBruijnBitPosition2[32] = 
  {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
  };
  return MultiplyDeBruijnBitPosition2[(uint32_t)(v * 0x077CB531U) >> 27];
}

void IsingMul(const float* b1, float* b2, float* hp, const unsigned int n, float gamma){
  //Originally this calculated b2 += H_G @ b1 by essentially using a fast walsh-hadamard transform
  //But the first part of it is compute heavy enough that we can put bandwidth limited calculations next to it for free
  //So now it does a full matrix multiply, setting b2 = (gamma*H_P - H_G) @ b1
  //Where H = gamma*H_P - H_G

  const unsigned int N = (1 << n);
  Vec16f a;
  Vec16f b;
  Vec16f H;
  int h = 16;
  //(1 << max_cache) should line up with cache size in some sense
  //Needs to be tuned for each machine ideally
  constexpr int max_cache = 15;

  //Use permutations for h<16 cases
  //We do enough computation here that we can load some out-of-cache data for free
  for (int i = 0; i<N; i+=16){
    a.load(b1+i);
    H.load(hp+i);
    b = a*H*gamma;

    b -= permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(a);
    b -= permute16<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(a);
    b -= permute16<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(a);
    b -= permute16<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(a);

    //For n larger than this, we can't fit vectors in cache anymore so these are
    //guaranteed cache misses
    if (n > max_cache){
      for (h = 1 << max_cache; h < N; h*=2){
        int temp = h&i ? -h : h;
        a.load(b1 + i + temp);
        b += a;
      }
    }

    b.store(b2+i);
  }

  //The vector can be kept in cache for these sizes
  int end = N < (1 << max_cache) ? N : 1 << max_cache;

  for (int i = 0; i<N; i+=16){
    b.load(b2+i);
    for (h = 16; h < end; h*=2){
      int temp = h&i ? -h : h;
      a.load(b1 + i + temp);
      b += a;
    }
    b.store(b2+i);
  }
  return;
}

class IsingOp
{
public:
  using Scalar = float;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  float gamma;

private:
  mutable Vector H_P;
  int N;
  int n_qubits;

public:
    // Constructor - takes the hp vector and gamma
  IsingOp(const Eigen::VectorXf& hp, float g) : H_P(hp), gamma(g) {
    N = H_P.size();
    n_qubits = log2(N);
  }

  int rows() const { return N; }
  int cols() const { return N; }

  // Matrix-free multiply y = H * x
  // Spectra calls this with pointers; must be const (Spectra API)
  void perform_op(const Scalar* x_in, Scalar* y_out) const {
    IsingMul(x_in, y_out, H_P.data(), n_qubits, gamma);
  }
};

int main(int argc, char* argv[]){
  //Arguments are number of spins, filename and number of problems
  //Last two are to allow for easier multi-threading, just start the program multiple times with different starts
  if (argc < 4) return -1;
  unsigned int n = atoi(argv[1]);
  unsigned int N = 1 << n;

  const int num_gaps = atoi(argv[2]);

  constexpr double gap_threshold = 1e-2;
  //A global minimum often has some local minima near it
  //So we enforce an exclusion radius around the true minimum
  constexpr double exclusion_radius = 0.1;

  //Chebyshev rounds up internally to the next power of 2
  //So this is actually is 64
  constexpr int cheb_samples = 31;
  
  std::string filename = "HardProblems" + std::to_string(n) + ".bin";

  //The polynomial degree is 1 less than the samples
  ProblemFile prob_file(filename, n, 63, num_gaps);

  int problems = atoi(argv[3]);

  std::cout << filename << " " << n << " " << problems << "\n";

  //Problem energy levels
  ArrayXf H_P(N);
  //This is used for calculating H_P, state(i,j) = sigma_z_i * sigma_z_j
  ArrayXXf state(n,n);

  //Our quantum register, can be entirely real here
  VectorXf psi(N);

  //Should use better RNG really
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{0.0, 1.0};

  //Ising problem parameters
  ArrayXXf J(n,n);
  ArrayXf h(n);

  for (int problem = 0; problem < problems; problem++){

    double E_0 = 0;
    double E_1 = 1e99;
    double E_max = 0;
    unsigned int E_loc = 0;

    J.setConstant(0);

    //Load J matrix
    int k = 0;
    for (int i = 1; i < n; i++){
      for (int j = 0; j < i; j++){
        J(i,j) = d(gen);
        k++;
      }
    }

    //Absorb 1/2 factor into J to follow paper
    J += J.transpose().eval();
    J /= 2;

    for (int i = 0; i < n; i++){
      h(i) = d(gen);
    }

    state.setConstant(1);
    psi.setConstant(1/sqrt(N));

    //The way we calculate energy levels is prone to floating point error so use a double
    //Start with the energy of the all -1s state
    double E = J.sum() - h.sum();
    H_P[0] = E;

    //In case first state is ground
    E_0 = E;
    E_max = E;
  
    //Use a grey code to efficiently evaluate all energies
    for (unsigned int i = 1; i < N; i++){
      unsigned int flip = log2(LSB(i));
      state.row(flip) *= -1;
      state.col(flip) *= -1;
      state(flip,flip) *= -1;
      E += 4*(J.row(flip)*state.row(flip)).sum() - 2*h(flip)*state(flip,flip);
      H_P[grey(i)] = E;

      //keep track of ground state
      if (E < E_0){
        E_1 = E_0;
        E_0 = E;
        E_loc = grey(i);
      } else if (E < E_1) {
        E_1 = E;
      }

      //keep track of highest state too
      if (E > E_max){
        E_max = E;
      }
    }

    //We want to shift H_P to reduce the spectral radius
    //This doesn't change the result but shortens calculations
    float h_gap = abs(E_0 - E_1);
    H_P -= (E_max + E_0)/2;
    float E_abs = (E_max - E_0)/2;
    //Now H_P has been calculated
    
    //We create a function which returns the energy gap at a given anneal fraction
    //Then pass that into Chebyshev fit
 
    IsingOp op(H_P.matrix(), 0);
    // Create a lambda that captures psi and op by reference

    auto gap_function = [&psi, &op, &h_gap](double x) -> double {
      //x goes from 1 to -1, which we map from 0 to 1
      double s = (1 - x) / 2;

      //avoid division by 0
      if (s == 1){return h_gap;}

      //We want (1-s)*H_G + s*H_P
      op.gamma = s/(1-s);

      //We want lowest 2 eigenvalues, using krylov subspace of size 10
      Spectra::SymEigsSolver<IsingOp> eigs(op, 2, 10);
      eigs.init(psi.data());
      int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 1000, (1-s)*1e-3);
      auto evecs = eigs.eigenvectors();
    
      //Update psi - it should be a good starting point for the next iteration
      psi = evecs.col(0).transpose() + evecs.col(1).transpose();
      auto evals = eigs.eigenvalues();
      double gap = abs(evals[1] - evals[0])*(1-s);

      return gap;
    };

    auto poly = Chebyshev<double>::fit(gap_function, cheb_samples, false);

    auto extrema = poly.deriv().roots();

    ArrayXf minima(num_gaps);
    ArrayXf min_locs(num_gaps);

    //Find minimum
    std::cout << "Minima are: ";

    for (int i = 0; i < num_gaps; i++){
      float min = 1e30;
      float min_loc = 0;
      for (const auto& a: extrema){
        //Make sure the extrema is on [-1,1] 
        if ((abs(a) < 1) and (abs(imag(a)) < 1e-10)){

          //Check we aren't in the exclusion zone of any of the other minima
          bool exclude = false;
          for (int j = 0; j < i; j++){
            if (abs(min_locs(j) - real(a)) < exclusion_radius){
              exclude = true;
            }
          }

          if (exclude){continue;}

          float temp = poly(real(a));//gap_function(real(a));
          if (min > temp){
            min = temp;
            min_loc = real(a);
          }
        }
      }
      minima(i) = gap_function(min_loc);
      min_locs(i) = min_loc;
      std::cout << "(" << minima(i) << "," << min_locs(i) << ")";
    }
    std::cout << "\n";
    if (minima.maxCoeff() < gap_threshold) {
      std::cout << "hard problem found\n\n";

      //Create ProblemInstance describing our problem
      ProblemInstance prob;
      prob.n = n;

      prob.params.resize(n * (n + 1) / 2);
      {
        size_t idx = 0;
        for (uint32_t i = 0; i < n; i++) {
          prob.params[idx++] = h(i);
          for (uint32_t j = 0; j < i; j++)
            prob.params[idx++] = J(i, j);
        }
      }

      prob.poly = poly;

      prob.num_minima = num_gaps;
      prob.minima.resize(2 * num_gaps);
      for (int i = 0; i < num_gaps; i++) {
        prob.minima[2*i]     = minima(i);
        prob.minima[2*i + 1] = min_locs(i);
      }
      //Save to disk
      prob_file.saveInstance(prob);
    }

  }
}
