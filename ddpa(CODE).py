import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics 
import warnings
from warnings import filterwarnings 
filterwarnings("ignore") 

# Set seaborn style
sns.set() 

# Load and merge data
df1 = pd.read_csv("D:/sem v/ddpA/calories (1).csv") 
df2 = pd.read_csv("D:/sem v/ddpA/exercise (1).csv") 
df = pd.concat([df2, df1["Calories"]], axis=1) 

# Drop unnecessary columns
df.drop(columns=["User_ID"], axis=1, inplace=True) 

# Create dummy variables for Gender
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Define plotting functions
def plot_distributions(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(20,15)) 
    plotnumber = 1 
    
    for column in numerical_cols: 
        if plotnumber <= 8: 
            ax = plt.subplot(3,3,plotnumber) 
            sns.histplot(df[column], kde=True) 
            plt.xlabel(column, fontsize=15) 
        plotnumber += 1 
    plt.show() 

def plot_correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10,10)) 
    sns.heatmap(df[numerical_cols].corr(), cmap='Blues', annot=True) 
    plt.show()

def predict(ml_model, X_train, X_test, y_train, y_test):
    model = ml_model.fit(X_train, y_train) 
    print(f'Training Score: {model.score(X_train, y_train):.4f}')
    
    y_prediction = model.predict(X_test) 
    
    # Calculate metrics
    r2_score = metrics.r2_score(y_test, y_prediction) 
    mae = metrics.mean_absolute_error(y_test, y_prediction)
    mse = metrics.mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    
    print(f'\nModel Evaluation Metrics:')
    print(f'R² Score: {r2_score:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}\n')
    
    # Display predictions
    print(f"Predictions: \n{y_prediction[:10]}...")  # Displaying first 10 predictions
    
    # Plot residuals
    plt.figure(figsize=(8,6))
    sns.histplot(y_test - y_prediction, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Prediction Error')
    plt.show()
    
    return model, r2_score, y_prediction

# Prepare features and target
X = df.drop(columns=["Calories"])
y = df["Calories"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("Dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}\n")

# Train and evaluate models
models = {
    'XGBoost': XGBRegressor(),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

best_model = None
best_score = -float('inf')
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}:")
    trained_model, score, predictions = predict(model, X_train, X_test, y_train, y_test)
    
    # Keep track of the best model
    if score > best_score:
        best_score = score
        best_model = trained_model
        best_model_name = name
        
# Final best model details
print(f"\nBest Model: {best_model_name}")
print(f"Best R² Score: {best_score:.4f}")
