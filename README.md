# spaceship_titanic_challenge
Data Science Workflow using a case study on spaceship titanic challenge on kaggle.

## Task
Predict which passengers are transported to an alternate dimension. 

## Requirements of challenge(Assignment)
1. EDA
2. Pre-processing
3. XAI(Explainable AI)

## 🚀 Data
The data was taken from the kaggle challenge(Spaceship Titanic).
PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
Age - The age of the passenger.
VIP - Whether the passenger has paid for special VIP service during the voyage.
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## 📊 Initial EDA
Performed comprehensive exploratory data preparation on the Spaceship Titanic dataset. Imported essential libraries (pandas, numpy, matplotlib, seaborn) and loaded both training and test datasets from GitHub URLs. Conducted initial data exploration by displaying sample rows, column names, dataset shapes, and detailed information about data types and null counts. Key achievements include statistical summaries of numerical features, uniqueness analysis  across all columns, and feature engineering by parsing the Cabin column into three separate components (Deck, Number, Side). Finally, performed missing value assessment for both datasets, establishing a solid foundation for subsequent modeling and visualization work.


## 📊 EDA Visualizations — Spaceship Titanic Train Data

This notebook provides visual insights from the training dataset of the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) challenge.  
The following plots explore relationships between features and the target variable `Transported`.

Spaceship Titanic - Exploratory Data Analysis (EDA)

This project explores the dataset for the Spaceship Titanic classification task — predicting whether a passenger was Transported to another dimension.


### 📊 Key Visual Insights

1. CryoSleep vs Transported

Passengers in CryoSleep=True were mostly transported.
Those not in CryoSleep were mostly not transported.
CryoSleep is a strong and clear indicator for prediction.

2. Spa vs VRDeck by Transported

Strong cluster of transported passengers with both expenses near-zero.
High-spending passengers were almost exclusively not transported.
Spending behavior on luxury services has a significant predictive signal.


## 📊Exploratory Data Analysis (EDA) – Test Dataset

We performed an in-depth EDA on the test dataset focusing on :
  1. Categorical Feature Distributions
  2. Numerical Feature Analysis
  3. Correlation Analysis
  4. Pairwise Relationships


## 🧼Preprocessing using Simple Imputation

### 1. Handling Missing Values

- Drops irrelevant columns like PassengerId, Cabin, and Name
- Applies Simple Imputation:
- Median for numerical features
- Mode for categorical features
  
### 2. Feature Encoding

- One-Hot Encoding for: HomePlanet, Destination, Cabin_Side
- Ordinal Encoding for: VIP, CryoSleep

### 3. Scaling

- Uses StandardScaler for all numerical features, excluding Transported in training



## 🚀 Models on Simple Imputation


### 📌 Model 1: XGBoost with Grid Search Optimization

* XGBoost classifier with a binary classification objective
* Hyperparameter tuning using GridSearchCV across key parameters including n_estimators, max_depth, learning_rate, subsample, and colsample_bytree on accuracy.
* Metrics : Accuracy, Classification Report, Confusion Matrix.
* Kaggle Submission score - 0.80383

#### XAI on Model 1: 

- SHAP :
  *  CryoSleep status is dominant — most decisive feature.
  *  Spending behaviors and planet of origin add predictive value, especially when combined with CryoSleep.
  *  Low-spenders and those from Europa, in CryoSleep, are more likely to be transported.
  *  High-spenders, VIPs, or those not in CryoSleep tend to be less likely.
    
- Feature Importance :
  *  CryoSleep dominates — likely a proxy for a major condition (e.g., safety, health, or status).
  *  Spending and origin features help but aren't as predictive.
  *  Age and destination matter very little, suggesting the model has found better signals elsewhere.
 

### 📌 Model 2: Random Forest with Grid Search Optimization

* Random Foest classifier
* Hyperparameter tuning using GridSearchCV across key parameters including n_estimators, max_depth, min_samples_split, and min_samples_leaf on accuracy.
* Metrics : Accuracy, Classification Report, Confusion Matrix.
* Kaggle Submission score - 0.79705
