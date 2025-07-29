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

#### 🔍 XAI on Model 1: 

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

* Random Forest classifier
* Hyperparameter tuning using GridSearchCV across key parameters including n_estimators, max_depth, min_samples_split, and min_samples_leaf on accuracy.
* Metrics: Accuracy, Classification Report, Confusion Matrix.
* Kaggle Submission score - 0.79705

#### 🔍 XAI on Model 2: 

- SHAP :
  *  CryoSleep is a dominant feature: being in CryoSleep is strongly associated with higher chance of being transported.
  *  Age interacts with CryoSleep: when passengers are not in CryoSleep, age has a more varied and less predictable influence on being transported.
  *  This supports the feature importance rankings: CryoSleep ranks highest, and Age is moderately important mostly through interactions rather than on its own.
    
- Feature Importance :
  *  CryoSleep and leisure/spending features like Spa and VRDeck are driving the predictions.
  *  Demographic and location features (Age, HomePlanet) are moderately important.
  *  VIP status and Destination are surprisingly less influential.
 
  
## 🧼Preprocessing using KNN Imputation

### 1. Missing Values Check
- Inspected missing data in both train and test sets.

### 2. KNN Imputation
- Applied OrdinalEncoder to temporarily convert categorical features.
- Used KNNImputer (k=5) to fill in missing values.
- Converted data back to categorical after imputation.

### 3. Encoding
- One-Hot Encoding for: HomePlanet, Destination, Cabin_Side
- Ordinal Encoding for: VIP, CryoSleep
- Ensured train/test sets have aligned columns post-encoding.
- Target Handling: Reattached and converted target variable Transported to integer.

### Train/Validation Split

Split the processed training data into 80% train / 20% validation.


## 🚀 Models on KNN Imputation


### 📌 Model 3: XGBoost with Grid Search Optimization

* XGBoost classifier with a binary classification objective
* Hyperparameter tuning using GridSearchCV across key parameters including n_estimators, max_depth, learning_rate, subsample, and colsample_bytree on accuracy.
* Metrics : Accuracy, Classification Report, Confusion Matrix.
* Kaggle Submission score - 0.79915

#### 🔍 XAI on Model 3: 

- SHAP :
  *  This SHAP summary plot shows that CryoSleep has the strongest impact on the model's output, with clear separation between low (blue) and high (red) values.
  *  Features like VRDeck, Spa, and HomePlanet_1.0 also influence predictions but to a lesser extent.
  *  Many other features have SHAP values clustered around zero, indicating they have minimal effect on the model’s decisions.
    
- Feature Importance :
  *  CryoSleep is the most influential feature, contributing the majority of the model’s predictive power.
  *  Most other features have minimal or no impact, with many contributing nothing at all.
  *  The feature encoding strategy likely needs refinement due to the presence of unusual fractional values.


### 📌 Model 4: Random Forest with Grid Search Optimization

* Random Forest classifier
* Hyperparameter tuning using GridSearchCV across key parameters including n_estimators, max_depth, min_samples_split, and min_samples_leaf on accuracy.
* Metrics: Accuracy, Classification Report, Confusion Matrix.
* Kaggle Submission score - 0.80102


  #### 🔍 XAI on Model 4: 

- SHAP :
  *  This SHAP interaction plot shows how Age and CryoSleep jointly affect your model’s predictions.
  *  The color-coded dots reveal patterns—such as younger passengers in CryoSleep contributing positively to prediction scores.
  *  It visualizes nuanced dependencies that single-feature analysis might miss.
    
- Feature Importance :
  *  CryoSleep and leisure/spending features like Spa and VRDeck are driving the predictions.
  *  Demographic and location features (Age, HomePlanet) are moderately important.
  *  VIP status and Destination are surprisingly less influential.
