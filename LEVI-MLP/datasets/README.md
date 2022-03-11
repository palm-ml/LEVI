## Introduction
The datafiles in this folder are used in the code of this project.

For N: #samples of trainset, M: #samples of testset (in classification task), D: #features, C: #labels

In recovery task, datafile of [dataset].plk with data structure as follows:
- **Info**: This part is a python dict which contains info of the dataset, the detailed data include:  `{'n_feature':D,'n_label':C, 'sparse':False}`
*(If the 'sparse' is True, the data of the features and labels is organized by the index of positive value, e.g. the label `[1,0,0,1,0]` is recorded as `[0,3]`);*
- **data**: This part includes the features and logical labels of the dataset, the structure of this part is: `{'data':np.array(N, D), 'label':np.array(N, C),'length':N}`
And datafile of [dataset]\_d.plk with a matrix of ground\_truth label distribution of np.array(N, C).

In classification task, ten the data file ([dataset].plk), the data structure is organized as follows:
- **Info**: This part is a python dict which contains info of the dataset, the detailed data include:  `{'n_feature':243,'n_label':6, 'sparse':False}`
- **traindata**：This part includes the features and labels of the trainset, the structure of this part is: `{'data':np.array(N, D), 'label':np.array(N, C),'length':N}` 
- **testdata**：This part includes the features and labels of the testset, the structure of this part is: `{'data':np.array(M, D), 'label':np.array(M, C),'length':M}` 

The convert.py can convert matlab datafile in `../matdata/` to plk datafile in this folder. 
