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

The accuracy score was about 0.72, meaning that the logistic regression model was only able to accurately predict 72% of the credit risk classifications correctly.

In addition to the `balanced_accuracy_score`, a `classification_report_imbalanced` was run to obtain precision and recall values for the logist regression model.

![nro_report](https://user-images.githubusercontent.com/107309793/194988708-5a4c2257-b4bc-4c47-9f87-e27d7a6b6a58.png)

The associated confusion matrix is as follows:

![nro_matrix](https://user-images.githubusercontent.com/107309793/194990991-17078f1a-2b78-47a5-938e-9390a0841820.png)

The precision, or the reliability of the model to classify correctly, is determined by the ratio TP/(TP + FP). For high credit risk (classified as 0) the precision is 58/(58 + 3894) which is approximately 0.01. For the low credit risk, it is 13224/(13224 + 29) which is approximately 1.00. The low precision value of 0.01 indicates that there are a lot of false positives (it predicted 3894 high risk candidates when there were truly only 58.) The high precision value of 1.00 likely indicates overfitting for the low-risk classification.

The recall, or the ability of the model to 

### Synthetic Minority Oversampling Technique (SMOTE)

### Cluster Centroid Undersampling

### SMOTE and Edited Nearest Neighbors (ENN) Combination Sampling

### Balanced Random Forest Classifier

### Easy Ensemble AdaBoost Classifier

## Summary
