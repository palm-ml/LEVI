## Introduction
The datafiles in this folder are not used in the code directly. But they can be converted to the datafiles in `../datasets/` by `../convert.py`to be used.

For N: #samples of trainset, M: #samples of testset (in classification task), D: #features, C: #labels

In recovery task, a datafile of [dataset]/[dataset]\_binary.mat should be prepared, with:
* N by D matrix 'features'
* N by C matrix 'labelDistribution'
* N by C 0/1 matrix 'logicalLabel'. 

In classification task, ten datafiles of [dataset]/[dataset]\_total\_[fold\_id].mat should be prepared, each with:
* N by D matrix 'train\_data'
* C by N 1/-1 matrix 'train\_target'
* M by D matrix 'test\_data'
* C by M matrix 1/-1 'test\_target'.
