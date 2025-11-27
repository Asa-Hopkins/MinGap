# Multistage-QW

## About The Project
This repository contains code for generating Ising problems with multiple small minimum gaps.

## Requirements
- [Eigen 3.4+](https://libeigen.gitlab.io/docs/)
- [Spectra](https://spectralib.org/)
- [Vector Class Libary 2](https://github.com/vectorclass/version2)
- The submodule [ApproxTools](https://github.com/Asa-Hopkins/ApproxTools) has some optional requirements 

## Getting Started
Once Eigen and VCL2 are installed, then clone everything with

`git clone --recurse-submodules https://github.com/Asa-Hopkins/MinGap/`

`cd MinGap`

Then build with \
`g++ -O3 -march=native MinGap.cpp -o MinGap`\
You may need to explicitly link Eigen by adding
`-I/usr/include/eigen3`

## Example Usage
The inputs to the program are: \
`n` - number of spins per problem \
`num_min` - the number of small minima the problem must have \
`problems` - how many problems to generate and check. Only the instances with small gaps are saved.

So the following would work \
`./MinGap 6 1 10000` \
and the resulting file contains 200 instances with small gaps.

Other parameters can be edited in the code directly

## To-Do
The main issue currently is the speed, for n = 18 it takes around 5s per problem to check the minimum gaps. Problems with two small minima only appear a few times per thousand problems, so it takes around half an hour to generate a hard problem with two minimum gaps currently.
Options to try improving this are:
GPU acceleration - tricky since I'd need to write a GPU based Lanczos solver
Better way of choosing problems - A suggestion I've seen is to generate hard problems of a smaller size, e.g n = 6, and then couple them together to create a bigger problem with more minima

## Contributing
I am open to contributions, discussions, criticism and feature requests.

## References
The code is largely copied from my other repository, MultiStageQW, with nothing really substantial added in terms of theory. For the full list of references, please check that repo and also the ApproxTools repo.
