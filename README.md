# Credit-Card-Fraud-Detection-Machine-Learning-Classification-Model
I developed a machine learning classification model to detect fraudulent credit card transactions using imbalanced datasets. The primary goal was to improve fraud detection accuracy while reducing false positives.


## **Project Objectives**
- Develop a machine learning model to detect fraudulent credit card transactions.
- Improve fraud detection accuracy while minimizing false positives.
- Handle imbalanced datasets using resampling techniques like SMOTE.
- Evaluate multiple machine learning classifiers for optimal performance.
- Deploy an efficient fraud detection system that can generalize well to real-world data.

## **Key Features**
- **Data Preprocessing:** Handling missing values, scaling, and feature selection.
- **Class Imbalance Handling:** Implemented Under-Sampling and SMOTE techniques (Borderline SMOTE).
- **Multiple ML Models:** Logistic Regression, K-NN, Random Forest, SVM, and Neural Networks.
- **Ensemble Learning:** Used Voting Classifier, Stacking, and Bagging Techniques.
- **Performance Metrics:** Evaluation based on AUC-ROC, Precision-Recall, and F1-Score.
- **Hyperparameter Tuning:** Optimized parameters using GridSearchCV and RandomizedSearchCV.
- **Visualization:** Interactive data analysis and fraud distribution insights.

## **Data Preprocessing Details**
- **Handling Class Imbalance:**
  - Applied Under-Sampling to balance class distribution in the initial phase.
  - Implemented Borderline SMOTE to generate synthetic samples near decision boundaries.
- **Feature Engineering:**
  - Selected important features to improve model accuracy.
  - Scaled numerical features to normalize the dataset.
- **Data Augmentation:**
  - Generated synthetic fraud cases to improve training effectiveness.

## **Visualizations**
- **Class Distribution Histogram:** Showcased the imbalance in fraud vs. non-fraud transactions.
- **Feature Importance Graphs:** Identified key predictors of fraudulent activity.
- **ROC & Precision-Recall Curves:** Evaluated model performance and trade-offs.

## **Implemented Models & Trials**
- **Phase One Models:**
  - Logistic Regression, K-NN, and Ridge Regression (Logistic Regression performed best).
- **Phase Two Models & Trials:**
  - **Random Forest with Bagging:** Achieved 94% accuracy.
  - **Neural Network (Adam & SGD Solvers):** Accuracy ranged from 94-95%.
  - **Voting Classifier (Decision Tree + SGD):** Initial accuracy 89%, later improved.
  - **SVM:** Initial accuracy 89-90%, later optimized with preprocessing.
  - **Stacking Ensemble (SVM, Decision Tree, SGD):** Helped boost accuracy but lower than other methods.
  - **Final Voting Model (SGD, Decision Tree, SVM, Random Forest):** Best accuracy achieved at 97.4%.

## **Main Questions Answered by the Dashboard**
- What is the distribution of fraudulent vs. non-fraudulent transactions?
- How do different machine learning models compare in fraud detection performance?
- What features contribute most to fraud detection?
- How does class imbalance affect the classification performance?
- What is the impact of different resampling techniques on model accuracy?
- How does ensemble learning improve fraud detection?
- How well does the final model generalize to unseen data?

## **Data Sources & Description:**
- the link of used data https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
