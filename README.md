# Credit_Risk_Analysis
Module 17

## Overview of the Analysis
LendingClub, a peer-to-peer lending services company, wants to utilize machine learning to predict credit risk. The management team belives that machine learning can accurately identify good candidates for loans and therefore lower loan default rates overall. Due to the unbalanced nature of the credit risk classification, different training techniques are needed. The purpose of this analysis was to employ several models and algorithms to a credit card dataset and to evaluate the performance of said models. This analysis involves resampling (oversampling, undersampling, and a combinatorial approach) and then later utilizes two different machine learning models (`BalancedRandomForestClassifier` and `EasyEnsembleClassifier`). The resampling and machine learning models are all compared in their ability to predict credit risk.

## Results

The credit card dataset was cleaned, split into features (all of the dependent data; everything except for `loan_status`) and target (the independent variable which assesses credit risk: `loan_status`), and then split further into training and testing groups. The majority of the target data shows scenarios with a low credit risk (all values of 1) while only a small portion of the target data shows scenarios with a high credit risk (values of 0). This is seen when checking the overall balance of the the target values in the code below:

![y_value_counts](https://user-images.githubusercontent.com/107309793/194986993-f62fd471-9cf5-40c2-9e41-8c148fb48de8.png)

This shows that only 347 datapoints are high credit risk while the other 68,470 datapoints are low credit risk. Due to this highly imbalanced dataset, resampling techniques were employed to balance the two classifications. **Logistic regression** was tested following these sampling techniques. The results are described below.

Note: Due to the high variation in value ranges between features, the dataset was scaled following resampling and prior to running through the machine learning models.

### Resampling

#### Naive Random Oversampling

In this case, instances of the low credit risk class were selected randomly and added to the training set until the high risk and low risk class were balanced. A `balanced_accuracy_score` was used to compute accuracy; this takes into account that this dataset was initially imbalanced and subsequently adjusted.

![nro_accuracy](https://user-images.githubusercontent.com/107309793/194987665-15d29c8d-881a-4b68-8fac-f1cc32b210db.png)

The accuracy score was about 0.72, meaning that the logistic regression model was only able to accurately predict 72% of the credit risk classifications correctly.

In addition to the `balanced_accuracy_score`, a `classification_report_imbalanced` was run to obtain precision and recall values for the logist regression model.

![nro_report](https://user-images.githubusercontent.com/107309793/194988708-5a4c2257-b4bc-4c47-9f87-e27d7a6b6a58.png)

The associated confusion matrix is as follows:

![nro_matrix](https://user-images.githubusercontent.com/107309793/194990991-17078f1a-2b78-47a5-938e-9390a0841820.png)

The precision, or the reliability of the model to predict the actual likelihood of high or low credit risk overall, is determined by the ratio TP/(TP + FP). For high credit risk (classified as 0) the precision is 58/(58 + 3894) which is approximately 0.01. For the low credit risk, it is 13224/(13224 + 29) which is approximately 1.00. The incredibly low precision value of 0.01 indicates that there are a lot of false positives (it predicted 3894 high risk candidates when there were truly only 58.) The high precision value of 1.00 likely indicates overfitting for the low-risk classification. A positive classification for high credit risk is very unlikely to be true in this case.

The recall, or the ability of the model to classify high-risk/low-risk correctly is determined by the ratio TP/(TP + FN). For high credit risk the sensitivity is 58/(58 + 29) which is approximate 0.67. The low sensitivity for the high risk classification indiates a lot of false negatives (it incorrectly categorized 29 people as low-risk when they were actually high-risk). The low credit risk sensitivy was 13224/(13224 + 3894) which is about 0.77. This higher sensitivity means there were less incorrectly categorized people as high-risk when they were actually low-risk.

#### Synthetic Minority Oversampling Technique (SMOTE)

In this case, the number of high credit risk instances was also increased, however these instances weren't replicated (as was done to increase the minority population in naive random oversampling), they were interpolated; the values were created based on neighboring values.

The balanced accuracy score was as follows:

![smote_accuracy](https://user-images.githubusercontent.com/107309793/195218988-f90498ed-c6b9-446b-89bf-86e9a4930308.png)

Again - the accuracy score was about 0.72, meaning that the logistic regression model was only able to accurately predict 72% of the credit risk classifications correctly. This is not an improved accuracy from the naive random oversampling method.

Here is the `classification_report_imbalanced` and confusion matrix:

![smote_report](https://user-images.githubusercontent.com/107309793/195219213-d248dd3a-afc2-4309-a4dd-573cf7ab4dab.png)

![smote_matrix](https://user-images.githubusercontent.com/107309793/195219222-56bcfc67-8d01-47b4-84b4-860e1fd21d59.png)

The classification report looks identical to the report with naive random oversampling; precision of 0.01 for high credit risk, 1.00 for low credit risk, a recall of 0.67 for high, and 0.77 for low risk. The same failings of the logistic regression model described above apply here also.

Interestingly there is a small difference seen in the confusion matrix for SMOTE. The number of false positives (model predicted 0 when the test was actually 1) went up compared to the naive random sampling method (from 3894 to 3918). This indicates that the precision was even worse in this case; the model is even less likely to classify high risk (goes down to 0.0146 from 0.0147). Lastly, the true negatives (model predicted 1 correctly) went down (from 13224 to 13200) indicating that the sensitivity for low credit risk decreases slightly (goes down to 0.771 from 0.772). This model is worse at accurately classifying low credit risk.

#### Cluster Centroid Undersampling

In cluster centroid undersampling, the algorithm determines low credit risk clusters within the dataset (data points that reside closely together) and then generates centroids (synthetic data points) which represent the whole cluster. The number of low credit risk centroids is decreased to the size of the high credit risk class. Logistic regression was then applied to this dataset again.

The balanced accuracy score for is as follows:

![ccu_accuracy](https://user-images.githubusercontent.com/107309793/195724127-09fc8878-5f77-49e5-b06b-80f3349de179.png)

The overall accuracy went down from the oversampling examples (down to 69% from 72%). This indicates that more low/high credit risk datapoints were classified incorrectly.

Here is the `classification_report_imbalanced` and confusion matrix:

![ccu_report](https://user-images.githubusercontent.com/107309793/195727459-fa39dd2a-4154-40ae-90d2-f64875b8b2f1.png)

![ccu_matrix](https://user-images.githubusercontent.com/107309793/195727463-e9eb932e-2603-45a9-a595-5c5733f35563.png)

After undersampling, the matrix shifts quite a bit. The logistic regression model predicts high credit risk incorrectly more often (11765 compared to 3894 and 3918) and low credit risk about the same as before (27 compared to 29 and 29). Despite the shift in the number of false negatives and false positives, the precision does not change significantly (0.01 for high credit risk and 1.00 for low credit risk).

It is noticeable that the sensitivity for high credit risk improves slightly (from 0.67 to 0.69) and the low credit risk sensitivity drops also (down to 0.69 from 0.77 in both oversampling techniques). This results in a rather balanced sensitivity for both high and low credit risk.

#### SMOTE and Edited Nearest Neighbors (ENN) Combination Sampling (SMOTEENN)
The final resampling technique employed to a logistic regression model was SMOTEENN. This combines the SMOTE and ENN algorithms. In this case, the high credit risk class was oversampled via synthetically generated datapoints. This is then followed by an undersampling where the two credit classes overlap.

Here is the new balanced accuarcy score with this technique:

![SMOTEENN_accuracy](https://user-images.githubusercontent.com/107309793/195726765-d166630a-79ad-4e46-9b48-a012edaa13ad.png)

The accuracy drops even further to 59% overall. High credit risk and low credit risk classification has a high chance of being incorrect with this logistic regression model.

Here are the rest of the results:

![SMOTEENN_matrix](https://user-images.githubusercontent.com/107309793/195727972-c6157a15-f917-4460-a682-6ce3fdef5bb5.png)

![SMOTEEN_report](https://user-images.githubusercontent.com/107309793/195727977-35c7d300-2413-4f1b-809f-cbfd12834dd2.png)

After undersampling, the matrix shifts quite a bit. The logistic regression model predicts high credit risk incorrectly more often (11765 compared to 3894 and 3918) and low credit risk incorrectly less often (15 compared to 29 and 29). Despite the shift in the number of false negatives and false positives, the precision does not change significantly (0.01 for high credit risk and 1.00 for low credit risk).

It is noticeable that the sensitivity for high credit risk improves significantly (from 0.67 to 0.83); however, this is not without a drop in sensitivity for low credit risk (down to 0.35 from 0.77 in both oversampling techniques).

### Ensemble Machine Learning

In the next two analyses, two different machine learning models were applied to the imbalanced high/low credit risk dataset. The dataset was scaled prior to employing the algorithms.

#### Balanced Random Forest Classifier
In this analysis, a type of random forest classifier is run: `BalancedRandomForestClassifier`. This utilizes a decision tree to determine high credit risk and low credit risk from the dataset features, but splits this up into many small decision trees. This classifier takes into account some random undersampling.

The `balanced_accuracy_score` is as follows:

![brf_accuracy](https://user-images.githubusercontent.com/107309793/195729739-9bce40bb-0ad4-4e0e-ad7b-7625e5d77d0e.png)

Up to this point, the `BalancedRandomForestClassifier` generates the highest accuracy score (about 77%). This is up from 72% compared to the two oversampling techniques. This model does a better job at classifying high and low risk overall.

Here is the `confusion_matrix` and the imbalanced classification report:

![brf_matrix](https://user-images.githubusercontent.com/107309793/195731209-c713e7e1-b1c6-4ba3-90de-1846edd462d1.png)

![brf_report](https://user-images.githubusercontent.com/107309793/195731217-fb86b23d-35fc-4f40-a658-498894b50886.png)

This model predicted 1958 false positives (for high credit risk) and 30 false negatives (for low credit risk). These are both improvements compared to all of the other models up to this point. Despite this fact, the precision still remains overfitted to low credit risk (0.03 for high and 1.00 for low).

The recall values shift in favor of low risk credit at 0.89. This indicates that a consumer that is classified as low credit risk is more likely to be classified correctly. There's sensitivity for high risk credit is still at a similar value as previous models (0.66 in this case).

#### Easy Ensemble AdaBoost Classifier
In this last analysis, a form of adaptive boosting is applied to the dataset: the `EasyEnsembleClassifier`. It evaluates errors as the model is trained which minimizes errors in subsequent models. The goal is to minimize error overall in determining a consumer's classification.

Here is the accuracy score:

![ee_accuracy](https://user-images.githubusercontent.com/107309793/195733336-533aa60c-7a55-4a29-8807-d4f695124b18.png)

This model has the highest accuracy at 92%.

Here is the confusion matrix and report:

![ee_matrix](https://user-images.githubusercontent.com/107309793/195733378-a25b9b87-4f57-44af-8765-0067726b959c.png)

![ee_report](https://user-images.githubusercontent.com/107309793/195733389-57174b4d-2da5-4580-a12c-a2e909667c04.png)

Looking at the matrix, the false positives for high credit risk and false negatives for low credit risk were reduced down to their smallest values compared to the other models (994 and 8 respectively). This results in a precision improvement for high credit risk (0.07).

The sensitivity for both high and low credit risk also improves signifantly (0.91 and 0.94).

## Summary

Here's a succinct summary of all of the results from every machine learning model (accuracy, precision for high credit risk & precision for low credit risk, recall for high credit risk & recall for low credit risk):

**Resampling**

- Naive Random Oversampling: 0.72, 0.01 & 1.00, 0.67 & 0.77
- SMOTE: 0.72, 0.01 & 1.00, 0.67 & 0.77
- Cluster Centroid Undersampling: 0.69, 0.01 & 1.00, 0.69 & 0.69
- SMOTEENN: 0.59, 0.01 & 1.00, 0.83 & 0.35

**Ensemble Machine Learning**

- Balanced Random Forest Classifier: 0.77, 0.03 & 1.00, 0.66 & 0.89
- Easy Ensemble AdaBoost Classifier: 0.92, 0.07 & 1.00, 0.91 & 0.94

In the scenario of evaluating whether a candidate is high credit risk or low credit risk for distributing a loan, it makes the most sense to minimize false negatives (predicting that a candidate is low credit risk when they are actually high credit risk). This means that the most optimal model will have a high precision for low credit risk (TN/(TN+FN)) and a high sensitivity for high credit risk (TP/(TP + FN)). Due to the imbalanced nature of this dataset, all of the models have a high precision for low credit risk (1.00). There is one model that has the highest sensitivity for high credit risk; the **Easy Ensemble AdaBoost Classifier** (0.91). This model also has the highest accuracy (0.92). This model would be the best choice for predicting high credit risk and minimizing false negatives. This model also minimizes false positives which will prevent too many people from being classified as high risk when they aren't.

