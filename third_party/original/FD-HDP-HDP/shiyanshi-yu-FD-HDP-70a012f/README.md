# FD-HDP
1. Project Overview
This project presents a Feature Disentanglement based Heterogeneous Defect Prediction (FD-HDP) method. It aims to address issues like data sparsity and feature heterogeneity in cross-project defect prediction. By disentangling features into domain-related and domain-independent ones, and integrating adversarial learning and feature reconstruction, it improves the defect prediction performance of target projects.
2. Project Structure
project/
├── dataset/
├── data_preprocessing.py
├── evaluate.py
├── feature_disentanglement.py
├── loss_functions.py
├── main.py
├── prediction_layer.py
├── train.py
├── README.txt
3. Method of Application
(1) Install Dependencies
pip install torch numpy pandas scikit-learn scipy imbalanced-learn
(2) Prepare the Dataset
Place the dataset file in the "dataset/" folder.
(3) Run the main program
python main.py
