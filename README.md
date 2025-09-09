**Calorie Brunt Prediction Using ML Algorithms**



This project predicts calories burned based on exercise data using multiple machine learning models.
It demonstrates data preprocessing, exploratory data analysis (EDA), model training, and evaluation in a structured workflow.

📂 Dataset

The project uses two datasets:

Calories Dataset (calories.csv) – contains calories burned information.

Exercise Dataset (exercise.csv) – contains exercise details like duration, heart rate, and more.

These datasets are merged, with unnecessary columns removed, and Gender converted to dummy variables for modeling.

⚙️ Features
🔎 Exploratory Data Analysis

Visualize distributions of numerical features.

Plot a correlation heatmap to understand relationships between features.

Example EDA Screenshots:

![Distribution Plot](./screenshots/distribution_plot.png)
![Correlation Heatmap](./screenshots/correlation_heatmap.png)

🧠 Predictive Modeling

Models trained:

XGBoost Regressor

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Evaluation Metrics for each model:

Model	R² Score	MAE	MSE	RMSE
XGBoost	0.9987	1.5526	5.2755	2.2969
Linear Regression	0.9656	8.4791	138.1241	11.7526
Decision Tree	0.9923	3.5110	30.7877	5.5487
Random Forest	0.9977	1.8182	9.4009	3.0661

Sample Predictions (first 10 values):

XGBoost: [197.06581, 70.867226, 196.99498, 16.840124, 72.875145, 23.09963, 5.074159, 147.85599, 255.69847, 6.907859]

Linear Regression: [198.81182363, 80.43555305, 194.40940033, 17.39285622, 78.9692843, 15.28475163, -0.3413037, 146.15851941, 209.87196487, -1.10270603]

Decision Tree: [194., 75., 204., 17., 72., 24., 5., 148., 253., 7.]

Random Forest: [196.87, 67.01, 197.11, 16.97, 73.24, 23.44, 5.26, 146.07, 256.74, 6.88]

Best Model: XGBoost
Best R² Score: 0.9987

📊 Visualizations

Distribution of Features

Correlation Heatmap

Residuals Plot for Best Model

(Include your screenshots like this)

![Residuals Plot](./screenshots/residuals_plot.png)

🚀 Getting Started
🔧 Requirements
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

▶️ Run the Script
python ddpa(CODE).py


This will:

Load and merge datasets.

Perform EDA with distribution plots and correlation heatmap.

Train multiple regression models and evaluate performance.

Display predictions, residuals, and the best-performing model.

👩‍💻 Tech Stack

Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost


🙌 Contributors

Thrisha Reddy J



