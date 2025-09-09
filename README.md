**Calorie Brunt Prediction Using ML Algorithms
**
This project predicts the calories burned based on exercise data using multiple machine learning models. The goal is to provide a data-driven way to estimate calorie expenditure during workouts.

It uses Python, Pandas, Seaborn, and Scikit-learn to preprocess data, perform exploratory data analysis (EDA), and build predictive models.

📂 Dataset

The project uses two datasets:

Calories Dataset (calories.csv): Contains calories burned information for users.

Exercise Dataset (exercise.csv): Contains exercise details such as duration, heart rate, and other parameters.

The datasets are merged to form the training data for machine learning models.

Note: The column User_ID is removed and Gender is converted to dummy variables for modeling.

⚙️ Features
🔎 Exploratory Data Analysis (EDA)

Visualize distributions of numerical features.

Generate a correlation heatmap to identify relationships between features.

🧠 Predictive Modeling

The project trains multiple regression models to predict calories burned:

XGBoost Regressor

Linear Regression

Decision Tree Regressor

Random Forest Regressor

For each model, the following metrics are computed:

R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Residuals of predictions are also visualized to understand model performance.

🚀 Getting Started
🔧 Requirements

Install the required Python libraries:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost

▶️ Run the Script
python ddpa(CODE).py


This will:

Load and merge the datasets.

Perform EDA with distribution plots and a correlation heatmap.

Train multiple regression models and evaluate their performance.

Display predictions and residual plots.

Print the best-performing model and its R² score.

📊 Visualizations

Distributions: Histograms of numerical features with KDE.

Correlation Heatmap: Shows relationships between features.

Residuals Plot: Distribution of prediction errors for each model.

👩‍💻 Tech Stack

Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

🙌 Contributors

Thrisha Reddy J
