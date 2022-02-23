# Credit_Risk_Analysis

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Overview**

The purpose of this project is to assist Fastlending, a peer-to-peer lending services company, in their implementation of machine learning to assess credit risk. Fastlending wants to use machine learning to predict credit risk, as the believe this will provide a quicker and more reliable loan experience. Furthermore, they also believe that machine learning will lead to a more accurate identification of good candidates for loans which will lead to lower default rates. We have been tasked by Fastlending to assist its lead data scientist in implementing this plan by building and evaluating several machine learning models or algorithms to predict credit risk. The techniques used to achieve this include resampling and boosting as part of the project. Once designed, the project called for us to rate their (the machine learning algorithms) analysis and performance to determine how well these models predict data.  

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Resources**

-Data Source: LoanStats_2019Q1.

-Software: Python 3.7.10; Visual Studio Code 1.64.2; Jupyter Notebook.

-Resources: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html; https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html; https://towardsdatascience.com/machine-learning-target-feature-label-imbalance-problem-and-solutions-98c5ae89ad0; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html; https://scikit-learn.org/stable/modules/linear_model.html; https://readthedocs.org/.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Module 17 - Challenge** 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Deliverable 1 - Use Resampling Models to Predict Credit Risk**  

**Deliverable 1 - Naive Random Oversampling**

![Deliverable 1 - Naive Random Oversampling](https://user-images.githubusercontent.com/92111396/155240559-e3e3292e-3a47-4ee2-aa74-b7f27616c360.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%201/Deliverable%201%20-%20Naive%20Random%20Oversampling.png


**Deliverable 1 - SMOTE Oversampling**

![Deliverable 1 - SMOTE Oversampling](https://user-images.githubusercontent.com/92111396/155240575-27f80bf4-6f4e-4f0f-b593-f4624f649388.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%201/Deliverable%201%20-%20SMOTE%20Oversampling.png


**Deliverable 1 - Undersampling**

![Deliverable 1 - Undersampling](https://user-images.githubusercontent.com/92111396/155240586-441cd8ea-cbef-4faf-b573-57f2931e46f2.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%201/Deliverable%201%20-%20Undersampling.png


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Deliverable 2 - Use the SMOTEENN algorithm to Predict Credit Risk**

**Deliverable 2 - Combination (Over and Under) Sampling (SMOTEEN)**

![Deliverable 2 - Combination (Over and Under) Sampling (SMOTEEN)](https://user-images.githubusercontent.com/92111396/155240517-350050c0-a6e8-47ca-a2b0-1c5d13c8cbb8.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%202/Deliverable%202%20-%20Combination%20(Over%20and%20Under)%20Sampling%20(SMOTEEN).png


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Deliverable 3 - Use Ensemble Classifiers to Predict Credit Risk**

**Balanced Random Forest Classifier**

![Deliverable 3 - Balanced Random Forest Classifier](https://user-images.githubusercontent.com/92111396/155334944-445916b0-ec1d-4c5a-a6cf-292d41e2b201.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%203/Deliverable%203%20-%20Balanced%20Random%20Forest%20Classifier.png


**Easy Ensemble AdaBoost Classifier**

![Deliverable 3 - Easy Ensemble AdaBoost Classifier](https://user-images.githubusercontent.com/92111396/155334981-6f969c33-0480-4ef0-8a5a-057d95431193.png)
https://github.com/mcbride249/Credit_Risk_Analysis/blob/main/Images/Deliverable%203/Deliverable%203%20-%20Easy%20Ensemble%20AdaBoost%20Classifier.png


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Deliverable 4 - Written Report on the Credit Risk Analysis**

## **Purpose**

The purpose of this challenge was to use machine learning to assist LendingClub, another peer-to-peer lending services company, to determine credit card risk. This challenge called for us to analyze a credit card dataset using a number of techniques and algorithms including oversampling the data using the “RandomOverSampler” and “SMOTE” algorithms, undersample the data using the “ClusterCentroids” algorithm, and use combinatorial approach of over and undersampling the data using the “SMOTEEN” algorithm. Finally, this project involved comparing two machine learning modules that reduce bias, the “BalancedRandomForestClassifier” and the “EasyEnsembleClassifier”, and evaluating their performance to predict credit risk.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Results**

**Accuracy Score**

-Naïve Random Oversampling Accuracy Score: 0.6405 = 64%

-SMOTE Oversampling Accuracy Score: 0.6585 = 66% 

-Undersampling Accuracy Score: 0.6585 = 66%

-Combination (Over and Under) Sampling Accuracy Score: 0.5442 = 54%

-Balanced Random Forest Classifier Accuracy Score: 0.7959 = 80%

-Easy Ensemble AdaBoost Classifier Accuracy Score: 0.9197 = 91%



**Precision Score**

-Naïve Random Oversampling Precision Score: 99%

-SMOTE Oversampling Precision Score: 99%

-Undersampling Precision Score: 99%

-Combination (Over and Under) Sampling Precision Score: 99%

-Balanced Random Forest Classifier Precision Score: 99%

-Easy Ensemble AdaBoost Classifier Precision Score: 99%



**Recall (Sensitivity) Score**

-Naïve Random Oversampling Sensitivity Score: 0.56 = 56%

-SMOTE Oversampling Sensitivity Score: 0.69 = 69%

-Undersampling Sensitivity Score: 0.40 = 40%

-Combination (Over and Under) Sampling Sensitivity Score: 0.57 = 57%

-Balanced Random Forest Classifier Sensitivity Score: 0.91 = 91%

-Easy Ensemble AdaBoost Classifier Sensitivity Score: 0.94 = 94%



**F1 Score**

-Naïve Random Oversampling F1 Score: 0.71 = 71%

-SMOTE Oversampling F1 Score: 0.81 = 81%

-Undersampling F1 Score: 0.56 = 56%

-Combination (Over and Under) Sampling F1 Score: 0.72= 72%

-Balanced Random Forest Classifier F1 Score: 0.95 = 95%

-Easy Ensemble AdaBoost Classifier F1 Score: 0.97 = 97%



**Definitions and Calculations of Scores**

Accuracy: the difference between its predicted values and actual values.

Precision: Precision = TP/(TP + FP) Precision is a measure of how reliable a positive classification is.

Sensitivity = TP/(TP + FN) Sensitivity is a measure of the probability of a positive test, conditioned on truly having the condition.

F1 = 2(Precision * Sensitivity)/(Precision + Sensitivity) A pronounced imbalance between sensitivity and precision will yield a low F1 score.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Summary**

Based on the above accuracy scores, we can see that AdaBoost Classifier machine learning model had the highest rate of accuracy with the ability to predict the correct values 91% of the time. Taken individually, the resampling models had similar accuracy scores falling between 64% and 66%, with the Smote oversampling and the undersampling techniques actually receiving the same accuracy score. While this may lead one to think that the combination of the two methods for testing would lead to a similar accuracy, we can see that this is not the case, with the combination machine learning model having the lowest accuracy score of all six learning models with a score of 54%. While there is not enough information to clearly state why the combination model had such a low score compared to the other models, we can speculate that a potential reason was due to the data being manipulated to the point where it was no longer accurate to the actual population. Based on these scores, the recommended learning models based on accuracy alone, would be Ensemble Classifiers, more specifically the AdaBoost Classifier machine learning model. 

The precision scores for all six machine learning models yielded the same precision score of 99%. This means that machine learning models can be relied upon to likely predict a positive classification 99% of the time. However, the precision score alone can tell us very little, and it must be coupled with the sensitivity of the score. The sensitivity scores effectively tell us how reliable in our prediction our tests are, that is to say, how fine-tuned or the probability of a positive test, conditioned on truly having the condition. Based on the above scores, it is evident Ensemble Classifiers were better tuned to correctly predict credit risk potential. The Balanced Random Forest Classifier had a recall score of 91%, while the Easy Ensemble AdaBoost Classifier had a recall score of 94%, once again ranking it as the most effective machine learning models for prediction. Compared to the resampling models, which had significantly lower scores, particularly the undersampling method which had the lowest recall score of 40%, it is evident that best machine learning model to be used based on sensitivity is the Easy Ensemble AdaBoost Classifier. 

The F1 scores of each model effectively tell us is there is a pronounced imbalance between sensitivity and precision; a pronounced imbalanced will yield a low F1 score. Based on this, we can again see that resampling methods fall short compared to the Ensemble Classifier machine learning models, with the undersampling technique having the lowest F1 score of 0.56 and with the Easy Ensemble AdaBoost Classifier having the largest F1 score of 0.97, thereby demonstrating the least disparity between sensitivity and precision. 

Based on the results and the subsequent analysis of the data, it is my recommendation that the Easy Ensemble AdaBoost Classifier machine learning model be adopted for use in predicting credit risk. It consistently had the highest scores, particularly in accuracy, precision, and sensitivity, and thus correctly made the correct predictions compared to the other models. 

