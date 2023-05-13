# Amex
We named our ML-based credit card fraudulent detection system as AegisX because aegis means protection or support. It depicts a shield. And like a shield our system will protect people from credit card frauds.
We build a real-time fraud detection system that utilizes a variety of machine learning algorithms. 
Our system analyzes past transaction data, discovers patterns, and learns to distinguish between authentic and fraudulent transactions.
Implentation steps of our system are as follow,
Data Collection: 
1)	We used ‘creditcard.csv’ imbalanced dataset from Kaggle to train our model.
2)	It contains 284807 rows and 31 columns.

Data Preparation:
1)	We used functions like isna().sum(), dropna() and fillna() to check, remove and replace missing data as s well as noise data.
2)	We used python libraries such as StandardScaler and RobustScaler to scale the skewed data.
3)	We used PCA python library to remove outliers. We also used it for dimensionality reduction.

Data Visualization:
1)	We used python libraries like matplotlib, seaborn and gridspec to visualize the data.
2)	We construct a number of charts such as histogram, count plot, boxplot, heatmap, displot to visualize our data.

Data Training:
To train the data we used machine learning algorithms such as 
1)	K Nearest Neighbor 
2)	Random Forest
3)	Naïve Bayes  
4)	Logistic Regression

Machine Learning Models Evaluation:
To evaluate our machine learning models we used
1)	Classification Report, which consists of precision, recall, f1 score, support, accuracy, macro avg, weighted avg.
2)	Confusion Matrix, which shows us the number of true positives(TP), true negatives(TN), false positives(FP) and false negatives(FN) for the set of prediction made by the model.

Testing of Machine Learning Model with New Data:

1)	We used credit_transdata.csv file from Kaggle to test our KNN and Naïve Bayes Machine Learning models.
2)	This dataset contains 179752 rows and 8 columns.
3)	KNN Machine Learning Model gave better accuracy than Naïve Bayes Machine Learning Model.

Model Deployment:
1)	We created a form using react.js so that users can interact with our Machine Learning Based Credit Card Fraudulent Detection System.
2)	We used fastapi and unicorn python libraries to create an API endpoint so that we can integrate our model with the form.

FUTURE SCOPE:

1)	We will try to deploy our project publicly in prototyping stage.
2)	We will create a functional website which will help people in credit card fraud.

Steps of the project:
1)	Data collection
2)	Data pre-processing
3)	Model training
I.	Using the clean data
II.	Using the imbalance data
4)	Evaluate the Machine Learning Models
5)	Test Machine Learning Models with new data
6)	Integrated the Machine learning Model with the form using the API
