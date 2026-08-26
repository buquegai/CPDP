% %
Created by: Zhiqiang Li
Email: lizq@snnu.edu.cn


This code is an implementation of the paper, which is described in:
Zhiqiang Li, Hongyu Zhang, Xiao-Yuan Jing, Juanying Xie, Min Guo and Jie Ren. 
"DSSDPP: Data Selection and Sampling based Domain Programming Predictor for Cross-project Defect Prediction" , 2022. 
It has submitted to the Journal of IEEE Transactions on Software Engineering under review.

Please kindly cite this paper if you would like to use the code.


%%%
This directory contains the following files:

DATASET.mat: there are 3 columns, the 1th column is the data of project, the 2nd column is the name of project, and the 3rd column is the random index to rum multiple time;

dataSelection: a file folder contains the implementations of maximum mean discrepancy (MMD), distributional characteristics based nearest neighbor selection (DCNNS), virtual classifier (VC), and Spearman correlation (CORR) algorithms; the replication package of Bellwether are available at https://goo.gl/jCQ1Le or by acquiring  the paper "TSE2019 Bellwethers: A Baseline Method for Transfer Learning"; 

*_ID.mat: the index of selected source project with data selection algorithm; "ID" is the index of sorted source based on the generated distance values, "numSources" is the number of selected sources by threshold in multi-source case; 

demo_DSSDPP.m: the main function to run the DSSDPP method;

DSSDPP.m: the implementation of Data Selection and Sampling based Domain Programming Predictor;

MPOS.m: them implementation of metric value permutation based over-sampling algorithm;

featureselection: a file folder contains some feature selection algorithms, which are downloaded from the URL of https://jundongl.github.io/scikit-feature/OLD/algorithms_old.html; 

DPP.m: the implementation of domain programming predictor;

utilities: a file folder contains some utility functions;

Please run demo_DSSDPP to obtain the prediction resutls.

Our running environment is MATLAB R2014a, 64bit operating system.
%%%


%%% NOTE %%%
The software is free for academic use only, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

