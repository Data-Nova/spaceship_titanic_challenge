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



