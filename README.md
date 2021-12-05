# MegBA
A High-Performance Distributed Library for Large-Scale Bundle Adjustment

MegBA has a novel end-to-end vectorised BA algorithm that can fully exploit the massive parallel cores on GPUs, thus speeding up the entire BA computation. 
It further has a novel exact distributed BA algorithm that can automatically partition BA problems, and solve BA sub-problems using distributed GPUs. 
The GPUs synchronise intermediate solving state using network-efficient collective communication, and the synchronisation is designed to minimise communication cost. 
MegBA has a memory-efficient GPU runtime and exposes g2o-compatible APIs. 
Experiments show that MegBA can out-perform state-of-the-art BA libraries (i.e., Ceres and DeepLM) by up to 47.6x and 6.4x respectively, in public large-scale BA benchmarks. 

* First Draft: https://arxiv.org/abs/2112.01349 (an updated version will be released by Dec 10)
* Code release (Beta version, expected Dec 06 2021)
* Code release (General version, TBD)
