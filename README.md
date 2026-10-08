#E-Commerce Customer Purchase Prediction Model


Name: Ancy John J.
Organization: Entri Elevate
Date: 07-10-2026

#1. Overview of Problem Statement


In an e-commerce website, not every visitor completes a purchase. Customer browsing behavior, page visits, time spent on different pages, visitor type, month, and other session-related information can provide useful insights into purchasing behavior.

The aim of this project is to develop a machine learning model that predicts whether an online customer is likely to make a purchase or not.



2. Objective


To develop an effective machine learning classification model for predicting customer purchase intention using online shopping session data and to identify the best-performing model among multiple classification algorithms.



3. Data Description

Dataset Name: Online Shoppers Purchasing Intention Dataset

Source: UCI Machine Learning Repository



Original Dataset Link:

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset


Original Dataset Size: 12,330 rows and 18 columns



Target Column: Revenue



True = Purchase


False = No Purchase


Features


Administrative


Administrative_Duration


Informational


Informational_Duration


ProductRelated


ProductRelated_Duration


BounceRates


ExitRates


PageValues


SpecialDay


Month


OperatingSystems


Browser


Region


TrafficType


VisitorType


Weekend



#4. Data Collection


The Online Shoppers Purchasing Intention Dataset was collected from the UCI Machine Learning Repository.



#The dataset was loaded into a Pandas DataFrame and inspected using:



head()


shape


info()


column information


data types


missing value checking


duplicate value checking


The original dataset contains 12,330 records and 18 columns.



##5. Data Preprocessing - Data Cleaning


#Missing Values


Missing values were checked using:

df.isnull().sum()


No missing values were found in the dataset.


Duplicate Values


Duplicate records were checked and 125 duplicate rows were found.


The duplicate rows were removed using:


df = df.drop_duplicates()




After removing duplicates, the working dataset contained 12,205 rows.


#Outlier Analysis


Outliers in numerical features were identified using the Interquartile Range (IQR) method.


Outliers were analyzed but were not automatically removed because they may represent genuine customer browsing behavior.
#

Skewness Analysis


Skewness of numerical features was analyzed. Several numerical features showed positive/right skewness.


The skewed values were analyzed but no automatic transformation was applied in the final model.


#36. Exploratory Data Analysis (EDA)


Exploratory Data Analysis was performed to understand the distribution of data and the relationship between features and the target variable.


#The following visualizations were used:


- Revenue distribution bar plot


- PageValues histogram


- PageValues vs Revenue boxplot


- Correlation heatmap


- VisitorType vs Revenue bar plot


- Month vs Revenue bar plot


- Weekend vs Revenue bar plot


- ProductRelated vs Revenue boxplot


EDA helped identify useful patterns in customer browsing behavior and purchase outcomes.


#7. Feature Engineering


The categorical features were converted into numerical form using One-Hot Encoding.


The following categorical features were encoded:


- Month


- VisitorType


The encoding was performed using:


pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)




After feature engineering, the dataset contained 26 input features.


#8. Feature Selection


Feature selection was performed using two approaches:


#Random Forest Feature Importance

Random Forest feature importance was used to identify important features affecting the prediction.


#SelectKBest

SelectKBest with ANOVA F-test (f_classif) was used to select the top 10 features.


#The selected features were:


- Administrative


- Administrative_Duration


- Informational


- ProductRelated


- ProductRelated_Duration


- BounceRates


- ExitRates


- PageValues


- Month_Nov


- VisitorType_Returning_Visitor


#39. Split Data into Training and Testing Sets


The dataset was divided into training and testing sets using an 80:20 ratio.


train_test_split(    X,    y,    test_size=0.2,    random_state=42,    stratify=y)




#Training Data

9,764 records


#Testing Data


2,441 records


Stratified splitting was used to maintain the class distribution in both training and testing datasets.


#310. Feature Scaling




Standardization was performed using StandardScaler.


Scaling was used for:


- Logistic Regression


- Support Vector Machine


- K-Nearest Neighbors


Tree-based models such as Decision Tree and Random Forest were trained without scaling.


#311. Build the ML Model


Five classification algorithms were implemented:


1. Logistic Regression


2. Decision Tree Classifier


3. Random Forest Classifier


4. Support Vector Machine (SVM)


5. K-Nearest Neighbors (KNN)


These models were selected to compare different classification approaches and identify the best-performing model.


##12. Model Evaluation


The classification models were evaluated using:


- Accuracy


- Precision


- Recall


- F1 Score


- Confusion Matrix


- ROC Curve


- ROC-AUC


#Model Comparison


Model	Accuracy	Precision	Recall	F1 Score


Logistic Regression	88.94%	77.18%	41.62%	54.08%


Decision Tree	86.03%	55.35%	55.50%	55.42%


Random Forest	90.82%	76.87%	59.16%	66.86%


SVM	89.47%	75.10%	48.95%	59.27%


KNN	87.34%	66.82%	37.96%	48.41%


#ROC-AUC


The Random Forest pipeline achieved:


ROC-AUC = 92.49%


13. Hyperparameter Tuning and Pipeline


Hyperparameter Tuning


GridSearchCV was used for Random Forest hyperparameter tuning.


The best parameters obtained were:


n_estimators = 200


max_depth = 20



##Pipeline


A machine learning pipeline was created using:


- SelectKBest


- Random Forest Classifier


The pipeline achieved:


Accuracy = 90.82%


ROC-AUC = 92.49%


#314. Save the Model


The trained Random Forest model was saved using Joblib.


#Saved files:


random_forest_model.pkl


feature_columns.pkl



These files are used by the Flask application for prediction.


##15. Test with Unseen Data


The saved Random Forest model was tested using an unseen customer record.


#The model generated:


- Purchase prediction


- Purchase probability


The Flask web application also accepts customer input through a frontend form and generates the prediction result.


##16. Interpretation of Results (Conclusion)


Five classification algorithms were tested and compared.


The Random Forest Classifier performed best overall with:


- Accuracy: 90.82%


- Precision: 76.87%


- Recall: 59.16%


- F1 Score: 66.86%


- ROC-AUC: 92.49%


Therefore, Random Forest was selected as the final model for the customer purchase prediction application.


The model can help identify whether an online visitor is likely to make a purchase based on browsing and session-related information.


Limitations

The dataset is imbalanced because the number of non-purchase sessions is much higher than purchase sessions. Therefore, accuracy alone should not be used to judge model performance.


##17. Future Work


Future improvements can include:


- Applying advanced class-balancing techniques


- Adding more customer and session-related features


- Retraining the model with newer data


- Exploring advanced machine learning and deep learning methods


- Improving the web application's user interface


- Deploying the application to a cloud platform


- Monitoring model performance after deployment
