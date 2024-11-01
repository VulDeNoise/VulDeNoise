# VulDeNoise

> A Generic Framework To Reduce Label Noises For Vulnerability Detection.

## Dataset

The Dataset we used in the paper:
Wenbo Wang, Tien N. Nguyen, Shaohua Wang, Yi Li, Jiyuan Zhang, and Aashish Yadavally, " DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection", in Proceedings of the 45th ACM/IEEE International Conference on Software Engineering (ACM/IEEE ICSE 2023), May 14-20, 2023. IEEE CS Press, 2023.
https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing

## Requirement

Please check all requirements in the requirement.txt

## Preprocess
1.Run preprocess file folder  ```raw_data_preprocess.py``` to get codes from MSR dataset.

2.Run preprocess/code_normalize file folder ```normalization.py``` to normalize the codes.

3.Use joern to generate PDG graphs, we use v1.1.172, please go to Joern's website: https://github.com/joernio/joern for more details on graph generation.

  We give py scripts in preprocess file folder ```joern_graph_gen.py```.You can refer to the required file.(.bin/.dot/.json)
  
4.Run preprocess file folder ```train_w2v.py``` to get trained w2v model.

5.Run preprocess file folder ```joern_to_devign.py``` to get the data required by the VD model.


## Differential Training
1. All codes in ```DT ``` file folder.
 
2. You need to modify ```data_loader/dataset.py ```.Pay attention to split the training set and test set(like train_set.txt/test.txt to provide data path)
 
3. You need to modify ```main.py ``` and ```trainer.py ``` like some input , output paths or models. First you need generate loss, please use train2 func, when you evaulate please use train func
 
4. Run ```main.py ``` to train or test.

## Outlier Detection
1.Please run outliers.py to detect noise samples, you need to modify some paths.

2.mykmeans.py provides a visualization with labeled clustering.




---

