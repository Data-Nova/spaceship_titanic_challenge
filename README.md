# spaceship_titanic_challenge
Data Science Workflow using a case study on spaceship titanic challenge on kaggle.

## Task
Predict which passengers are transported to an alternate dimension. 

## Requirements of challenge(Assignment)
1. EDA
2. Pre-processing
3. XAI(Explainable AI)

## Data
The data was taken from the kaggle challenge(Spaceship Titanic).

## Exploratory Data Analysis (EDA) – Test Data

In this project, we performed **comprehensive EDA** on the **test dataset** (from the Spaceship Titanic challenge).  
The goal was to understand data distributions, detect anomalies, and prepare for preprocessing and modeling.

### Key Steps in EDA:
1. **Basic Structure & Info**  
   - Checked data types, missing values, and overall shape.
   - Summarized categorical vs. numerical features.

2. **Univariate Analysis**  
   - **Histograms & Boxplots** for numerical features: `Age`, `RoomService`, `FoodCourt`, `Spa`, `VRDeck`.  
   - Detected skewness and outliers (especially in spending-related features).

3. **Correlation Analysis**  
   - Generated a **heatmap** for numerical features to examine relationships.
   - Observed weak correlations between spending features (RoomService, FoodCourt, Spa, VRDeck) and each other.

4. **Pairplots for Numerical Features**  
   - Created **pairwise scatter plots** for key continuous variables (`Age`, `RoomService`, `FoodCourt`, `Spa`, `VRDeck`).  
   - Excluded the target (`Transported`) since this is **test data**.

5. **Categorical Distributions**  
   - Analyzed frequency counts for key categorical columns (HomePlanet, CryoSleep, Destination, VIP).

### Final EDA Visualizations
- **Distribution plots** for age and spending features (RoomService, FoodCourt, Spa, VRDeck).  
- **Correlation heatmap** for numerical columns.  
- **Pairplots** for deeper exploration of multivariate relationships.  

These insights guided the **feature preprocessing** and **modeling** phases (encoding, scaling, handling missing values).
