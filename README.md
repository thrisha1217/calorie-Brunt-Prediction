**Calorie Brunt Prediction Using ML Algorithms**


This project predicts calories burned from exercise data using advanced machine learning models. The workflow demonstrates robust data preprocessing, exploratory data analysis (EDA), multiple model training, and thorough evaluation.

📂 Dataset
Calories Dataset (calories.csv): Contains calories burned per session.

Exercise Dataset (exercise.csv): Contains exercise features like duration, heart rate, etc.

The datasets are merged for a unified analysis, with unnecessary columns dropped and Gender converted to dummy variables.

⚙️ Features
🔎 Exploratory Data Analysis
Visualizes distributions for all major numerical features.
Linear Regression
Decision Tree Regressor
Random Forest Regressor

Evaluation Metrics
Model	R² Score	MAE	MSE	RMSE
XGBoost	0.9987	1.5526	5.2755	2.2969
Linear Regression	0.9656	8.4791	138.1241	11.7526
Decision Tree	0.9923	3.5110	30.7877	5.5487
Random Forest	0.9977	1.8182	9.4009	3.0661
Sample Predictions (first 10):
XGBoost: [197.07, 70.87, 196.99, 16.84, 72.88, 23.10, 5.07, 147.86, 255.70, 6.91]

Linear Regression: [198.81, 80.44, 194.41, 17.39, 78.97, 15.28, -0.34, 146.16, 209.87, -1.10]

Decision Tree: [194., 75., 204., 17., 72., 24., 5., 148., 253., 7.]

Random Forest: [196.87, 67.01, 197.11, 16.97, 73.24, 23.44, 5.26, 146.07, 256.74, 6.88]

📊 Visualizations
Residuals Plot for XGBoost (Best Model)
Most errors are around zero, confirming superior accuracy, low bias, and high predictive power.
![Residuals Plot - XGBoost](./Visualizations/residuals_plot_xgboost.png)


Residuals Plot for Linear Regression
Wider error dispersal with larger outliers, consistent with lower model performance on this nonlinear dataset.
![Residuals Plot - Linear Regression](./screenshots/residuals_plot_linear.png)


Residuals Plot for Decision Tree
Sharp peak with some error spread, indicating good fit but less precision than ensemble models.
![Residuals Plot - Decision Tree](./screenshots/residuals_plot_tree.png)


Residuals Plot for Random Forest
Tight clustering and minimal outliers—performance slightly below XGBoost but clearly superior to non-ensemble models.
![Residuals Plot - Random Forest](./screenshots/residuals_plot_forest.png)


🚀 Getting Started
🔧 Requirements
Install dependencies:

bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
▶️ Run the Script
Execute:

bash
python Calorie_Brunt_Code.py

This will:

Load and merge the datasets.

Perform comprehensive EDA (distribution plots, correlation heatmap).

Train regression models and evaluate their performance.

Display key predictions, residual plots, and highlight the best model based on R² score.

👩‍💻 Tech Stack
Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

🙌 Contributors
Thrisha Reddy J




