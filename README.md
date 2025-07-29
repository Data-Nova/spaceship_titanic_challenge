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


# 📊 EDA Visualizations — Spaceship Titanic Train Data

This notebook provides visual insights from the training dataset of the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) challenge.  
The following plots explore relationships between features and the target variable `Transported`.

Spaceship Titanic - Exploratory Data Analysis (EDA)

This project explores the dataset for the Spaceship Titanic classification task — predicting whether a passenger was Transported to another dimension.

📊 Visualizations Summary

1. Target Distribution

The target variable Transported is nearly balanced between True and False.

This implies that accuracy can be a reliable performance metric.

2. HomePlanet vs Transported

Passengers from Europa were more likely to be transported.

Earth had more passengers, but a lower transport rate.

Mars had the most balanced distribution between Transported classes.

3. CryoSleep vs Transported

Passengers in CryoSleep=True were mostly transported.

Those not in CryoSleep were mostly not transported.

CryoSleep is a strong indicator for prediction.

4. Age Distribution

Age is right-skewed; most passengers are between 20–40.

Fewer elderly passengers.

5. Correlation Heatmap

Transported shows weak negative correlations with:

RoomService, Spa, VRDeck

Positive correlations with none of the features.

Multicollinearity is not a concern.

6. Age vs RoomService by Transported

Most transported passengers had low RoomService expenses.

Non-transported ones spent more.

7. Age vs FoodCourt by Transported

Similar to RoomService; higher spending is associated with non-transported passengers.

8. Age vs ShoppingMall by Transported

No strong visible separation between classes.

Slightly higher Transported ratio with low spending.

9. Age vs Spa by Transported

Transported passengers spent significantly less on Spa.

10. Age vs VRDeck by Transported

Pattern similar to Spa; lower spending increases likelihood of being transported.

