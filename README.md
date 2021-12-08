# MegBA: A High-Performance and Distributed Library for Large-Scale Bundle Adjustment

This repo contains an official implementation of [MegBA](https://arxiv.org/abs/2112.01349).

MegBA is a fast and distributed library for large-scale Bundle Adjustment (BA). MegBA has a novel end-to-end vectorised BA algorithm which can fully exploit the massive parallel cores on GPUs, 
thus speeding up the entire BA computation. It also has a novel distributed BA algorithm that can automatically partition BA problems, 
and solve BA sub-problems using distributed GPUs. The GPUs synchronise intermediate solving state using network-efficient collective communication, 
and the synchronisation is designed to minimise communication cost. MegBA has a memory-efficient GPU runtime and it exposes g2o-compatible APIs. 
Experiments show that MegBA can out-perform state-of-the-art BA libraries (i.e., Ceres and DeepLM) by ~50x and ~5x respectively, in public large-scale BA benchmarks.


## Version
* 2021/12/06 Beta version released! It corresponds to this [paper](https://arxiv.org/abs/2112.01349).
* General version code release (Expected Dec 31 2021)
* memory-efficient version with implicit Hessian (TBD)
* analytical differential module, IMU factor, prior factor (TBD)

Paper:
* First Draft: https://arxiv.org/abs/2112.01349 (an updated version will be released by Dec 10)


## Quickstart

Dependencies:
- C++14
- CMake (>= 3.15)
- CUDA Toolkit (with thrust) https://developer.nvidia.com/cuda-downloads
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (MegBA Beta depends on a modified version of Eigen under third_party/. Eigen-free version will be provided in the general release.)
- NCCL (if you need Distributed features) https://developer.nvidia.com/nccl/nccl-download

You can also easily install all dependencies with script: [script](https://drive.google.com/file/d/154whcVH2VcJCYnTSlnfo_tbIIaQvSax3/view?usp=sharing)


Demo with BAL dataset:
* Download any pre.txt.bz2 file from BAL Dataset: https://grail.cs.washington.edu/projects/bal/ and uncompressed.
* Compile
  ```bash
  mkdir build
  cd build
  cmake ..
  make -j4 BAL_Double
  ```

* Run the demo (Venice-1778)
  ```bash
  cd examples
  ./BAL_Double --name=Venice --worldSize=2 --iter=100 --solver_tol=1e-1 --solver_refuse_ratio=1 --solver_max_iter=100 --tau=1e4 --epsilon1=1 --epsilon2=1e-10
  ```
  - worldSize: number of GPUs available
  - iter: the maximal number of LM iteration
  - epsilon: threshold in LM
  - solver_tol: tolerance of solver (distributed PCG solver)
  - solver_refuse_ratio: early stop for the solver
  - solver_max_iter: the maximal iteration of solver
  - tau: the initial region


## Notes for the practitioners
* Currently, MegBA implements automatic differentation only for generalizability. Please consider implementing your own analytical differentiation module.
* If you use devices without modern inter-device communication (i.e., NVLinks..), you might find the data transfer is the bottleneck.
* Empirically, we found it is necessary to customize the LM trust-region strategies and tune its hyper-parameters to further boost the performance. 


## Documentation
Under doc/  (Coming soon...)


## Collaborate with Us
Please check here for [MegBA's future plan](https://docs.google.com/document/d/1fHYuw_qRFHrBcGSeQ8Ld4y2wK9oxF0am3xA9r6veUwM/edit?usp=sharing).

If you are intereted in MegBA and want to collaborate, you can:
* Apply for an Internship at Megvii Research 3D, please send your resume to ur@megvii.com, with your expected starting date. (subject: 3D组CUDA实习生-Name) Unfortunately, now we are only able to host interns with work authorization in China. 
* As an external collaborator (coding), just fork this repo and send PRs. We will review your PR carefully (and merge it into MegBA).
* As an algorithm/novelty contributor, please send an email to MegBA@googlegroups.com.
* Any new feature request, you can send an email to MegBA@googlegroups.com as well. *Note that it is not guaranteed the requested feature will be added or added soon*


Contact Information:
* Jie Ren jieren9806@gmail.com
* Wenteng Liang wenteng_liang@163.com
* Ran Yan yanran@megvii.com
* Shiwen Liu lswbblue@163.com
* Xiao Liu liuxiao@foxmail.com

## BibTeX Citation
If you find MegBA useful for your project, please consider citing:
```
@misc{2021megba,
  title={MegBA: A High-Performance and Distributed Library for Large-Scale Bundle Adjustment}, 
  author={Jie Ren and Wenteng Liang and Ran Yan and Luo Mai and Shiwen Liu and Xiao Liu},
  year={2021},
  eprint={2112.01349},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

```


## License
MegBA is licensed under the Apache License, Version 2.0.

