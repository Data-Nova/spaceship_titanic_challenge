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

## Initial EDA
Performed comprehensive exploratory data preparation on the Spaceship Titanic dataset. Imported essential libraries (pandas, numpy, matplotlib, seaborn) and loaded both training and test datasets from GitHub URLs. Conducted initial data exploration by displaying sample rows, column names, dataset shapes, and detailed information about data types and null counts. Key achievements include statistical summaries of numerical features, uniqueness analysis  across all columns, and feature engineering by parsing the Cabin column into three separate components (Deck, Number, Side). Finally, performed missing value assessment for both datasets, establishing a solid foundation for subsequent modeling and visualization work.

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

## 🧪 Exploratory Data Analysis (EDA) on Train Data

A detailed EDA was conducted on the Spaceship Titanic training dataset to uncover relationships, distributions, and feature behaviors prior to modeling.

###  Target Variable (`Transported`)
- The `Transported` column is the target for prediction.
- The classes are approximately balanced, making it suitable for classification without major resampling.

###  Categorical Features
- `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, and parsed `Cabin` components (`Deck`, `Side`) were examined.
- `CryoSleep` showed a strong correlation: passengers in cryo sleep were more likely to be transported.
- `VIP` passengers had slightly lower transport rates.
- `HomePlanet` and `Destination` influenced transport probability.

###  Numerical Features
- Age had a mild trend: younger passengers had a higher chance of being transported.
- Spending features (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) showed meaningful patterns—non-spending passengers were more likely to be transported.
- Null values in these columns were treated as zeros, indicating no spending.

###  Correlation Analysis
- A correlation heatmap revealed:
  - Moderate positive correlation between different spending features.
  - Negative correlation between spending and the target in some cases (e.g., `Spa`, `VRDeck`).

###  Multivariate Visualizations
- 2D scatterplots and pairwise plots of numeric columns highlighted clusters and feature interactions.
- Color-coded plots showed class separation in feature combinations such as:
  - `Age` vs `RoomService`
  - `Spa` vs `VRDeck`
  - `ShoppingMall` vs `FoodCourt`


## Exploratory Data Analysis (EDA) – Test Data

In this project, we performed **comprehensive EDA** on the **test dataset** (from the Spaceship Titanic challenge).  
The goal was to understand data distributions, detect anomalies, and prepare for preprocessing and modeling.

