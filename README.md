# HEART-DISEASE
💓 Heart Disease Prediction using Machine Learning
This project is a comprehensive analysis and prediction model for heart disease using the UCI Heart Disease Dataset. It includes data preprocessing, visualization, exploratory data analysis (EDA), and training of multiple machine learning models to classify the presence of heart disease in patients.

🧰 Tools & Libraries Used
Pandas – for data loading and manipulation

NumPy – for numerical operations

Matplotlib & Seaborn – for data visualization

Scikit-learn – for machine learning models and evaluation

📁 Dataset
The dataset contains 303 rows and 14 features, including both patient information and medical data.
Key features include:

age, sex, cp (chest pain type), thalach (max heart rate), oldpeak, chol, etc.

target: Binary classification label (1 = presence of heart disease, 0 = absence)

Source: UCI Machine Learning Repository

📊 Exploratory Data Analysis (EDA)
The project includes:

Distribution of heart disease by gender.

Relationship between age and max heart rate.

Frequency of heart disease across chest pain types.

Correlation heatmap to understand relationships between features.

Visualizations helped generate intuitive insights like:

Younger patients tend to have higher max heart rates.

Certain chest pain types are strongly associated with heart disease.

Gender-wise distribution shows higher likelihood in females based on this dataset.

🔍 Data Preprocessing
Checked for missing values (none found).

Separated features (X) and target (y).

Performed train-test split using train_test_split().

🤖 Machine Learning Models Used
Three ML models were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Model performance was evaluated using:

Confusion Matrix

Precision, Recall, F1 Score

Cross-validation scores

ROC Curve

✅ Results
The model achieved decent classification performance with metrics such as:

High F1-score

Balanced precision-recall

Clear ROC curves

Further improvement is possible using hyperparameter tuning with GridSearchCV or RandomizedSearchCV.

📈 Visual Highlights
Bar plots to show gender vs disease prevalence.

Scatter plot for Age vs Max Heart Rate for disease prediction.

Heatmap showing correlation between features.

Histogram showing age distribution.

🧠 Conclusion
This project demonstrates how machine learning can be applied to healthcare for early prediction of heart disease. With more data and deeper analysis, models like these can help medical professionals make more informed decisions.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
Install required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Run the notebook:

Use Google Colab or Jupyter Notebook

Load heart.csv from your directory or drive

👤 Author
Munja Saideepak
Aspiring Data Scientist | Developer | ML Enthusiast
📧 saideepak.ui@gmail.com
🔗 LinkedIn • GitHub

📌 Note
This is a beginner-level end-to-end machine learning classification project. The model is not intended for clinical use but for educational purposes.

