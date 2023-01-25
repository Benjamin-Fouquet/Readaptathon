# Readaptathon   

IMT repository from the 2022 readaptathon in Rennes.  It includes:   
-Interpolation folder to pre-process the data   
-datamodules to covert open pose json files to tensor datasets to be fed to models   
-models for training and inference   
-test folder   
-visualisation folder includes fonction to facilitate the visualisation of data from the datamodule 
  
Data are expected to be open pose outputs when fed to datamodules. Irrelevant pose data are to be discarded prior to training.  
  
Models are derived from the following literature: 
  
![image](https://user-images.githubusercontent.com/77781338/214563583-161fa476-e151-4c49-8b20-4f0e393936d3.png) 
  
Salami et al.:Using Deep Neural Networks for Human Fall Detection Based on Pose 
  
![image](https://user-images.githubusercontent.com/77781338/214563665-ddd48aae-f974-40c6-8ea0-98b79961f017.png) 
  
Yann et Al.: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition  




