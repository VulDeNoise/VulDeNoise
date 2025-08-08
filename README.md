# VulDeNoise

> A Generic Framework To Reduce Label Noises For Vulnerability Detection.

## Dataset

The new Big-Vul Dataset we used in the paper:
Wenbo Wang, Tien N. Nguyen, Shaohua Wang, Yi Li, Jiyuan Zhang, and Aashish Yadavally, " DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection", in Proceedings of the 45th ACM/IEEE International Conference on Software Engineering (ACM/IEEE ICSE 2023), May 14-20, 2023. IEEE CS Press, 2023.
https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing

And the old version is: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing

The FFmpeg+QEMU Dataset: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit

## Requirement

Please check all requirements in the requirement.txt

## Preprocess
1.Run preprocess file folder  ```raw_data_preprocess.py``` to get codes from MSR dataset.

2.Run preprocess/code_normalize file folder ```normalization.py``` to normalize the codes.

3.Use joern to generate PDG graphs, we use v1.1.172, please go to Joern's website: https://github.com/joernio/joern for more details on graph generation.

  We give py scripts in preprocess file folder ```joern_graph_gen.py```.You can refer to the required file.(.bin/.dot/.json)
  
4.Run preprocess file folder ```train_w2v.py``` to get trained w2v model.

5.Run preprocess file folder ```joern_to_devign.py``` to get the data required by the VD model.


## Multi-View Learning
1. All codes in ```VulDenosie ``` file folder.
 
2. For vulnerability detection model training, you need to use ```train_model.py ```. 
 
3. For generating loss vectors, you can use ```loss_vec.py ```
   
4. We also provied a script for 5-fold cross validation ```5_cross_val.py ```
 

## Outlier Detection
1.For outlier detection, you can use ```outlier.py ```.

2.We also provie a script ```denoising_metrics.py ``` to get the denoising performance and remained noises.

3.You can combine ```denoising_metrics.py ``` and ```train_model.py ``` to get new VD model detection performance.

## Cases
We provide some noisy cases in ```invalid_modification_noises.txt```


---

