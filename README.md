# CAE-Transfer
In this project, a convolutinal auto-encoder based unsupervised learning and its transfer learning are built  
  
## Citation
The code is a recurrence privately, and the reproduced paper is:  
M. Xia, H. Shao, Z. Huang, Z. Zhao, F. Jiang and Y. Hu,   
"Intelligent Process Monitoring of Laser-Induced Graphene Production With Deep Transfer Learning",  
IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-9, 2022, Art no. 3516409,   
doi: 10.1109/TIM.2022.3186688.   
  
## Environment   
The case ideally requires:   
Python>=3.8   
keras>=2.6.0   
numpy>=1.19.5   
Scikit-learn>=0.24.1   
matplotlib>=3.3.4   
 
## Nets.py 
The code is used for buliding the network to be trained, including Convolutional Auto-Encoder (CAE),   
Enhanced Convolutional Neural Network (ECNN) which is same with the paper. The every models will be   
built and compile, to test with default parameters.  
   
## Transfer.py 
The code is used for unsupervised learning and transfer learning according to the paper. The dataset   
is not provided because of some reasons, but the dataset form adapted this code is given:  
1) the dataset should be composed of picture;  
2) the directory of dataset should submit to the following structure, or correcting the code:   
dataset   
└───source   
|---└───unlabeled   
|---|---└───Train   
|---|---└───Valid   
|---|---└───Test   
|---└───labeled   
|-------└─── Train   
└───target   
----└───Train   
----└───Valid   
----└───Test   
   
## Test_and_Plot.py  

