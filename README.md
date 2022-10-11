# Credit_Risk_Analysis
Module 17

## Overview of the Analysis
LendingClub, a peer-to-peer lending services company, wants to utilize machine learning to predict credit risk. The management team belives that machine learning can accurately identify good candidates for loans and therefore lower loan default rates overall. Due to the unbalanced nature of the credit risk classification, different training techniques are needed. The purpose of this analysis was to employ several models and algorithms to a credit card dataset and to evaluate the performance of said models. This analysis involves resampling (oversampling, undersampling, and a combinatorial approach) and then later utilizes two different machine learning models (`BalancedRandomForestClassifier` and `EasyEnsembleClassifier`). The resampling and machine learning models are all compared in their ability to predict credit risk.

## Results

The credit card dataset was cleaned, split into features (all of the dependent data; everything except for `loan_status`) and target (the independent variable which assesses credit risk: `loan_status`), and then split further into training and testing groups. The majority of the target data shows scenarios with a low credit risk (all values of 1) while only a small portion of the target data shows scenarios with a high credit risk (values of 0). This is seen when checking the overall balance of the the target values in the code below:

![y_value_counts](https://user-images.githubusercontent.com/107309793/194986993-f62fd471-9cf5-40c2-9e41-8c148fb48de8.png)

This shows that only 347 datapoints are high credit risk while the other 68,470 datapoints are low credit risk. Due to this highly imbalanced dataset, resampling techniques were employed to balance the two classifications. **Logistic regression** was tested following these sampling techniques. The results are described below.

### Naive Random Oversampling

In this case, instances of the low credit risk class were selected randomly and added to the training set until the high risk and low risk class were balanced. A `balanced_accuracy_score` was used to compute accuracy; this takes into account that this dataset was initially imbalanced and subsequently adjusted.

![nro_accuracy](https://user-images.githubusercontent.com/107309793/194987665-15d29c8d-881a-4b68-8fac-f1cc32b210db.png)

The accuracy score was about 0.72, meaning that the 

### Synthetic Minority Oversampling Technique (SMOTE)

### Cluster Centroid Undersampling

### SMOTE and Edited Nearest Neighbors (ENN) Combination Sampling

### Balanced Random Forest Classifier

### Easy Ensemble AdaBoost Classifier

## Summary
