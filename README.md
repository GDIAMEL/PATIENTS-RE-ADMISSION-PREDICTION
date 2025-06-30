# PATIENTS-RE-ADMISSION-PREDICTION


🏥 Patient Readmission Prediction (within 30 days) – Step-by-Step Project Guide
⚙️ Step 1: Set Up Environment
bash
Copy
Edit
# If using a local machine (skip if using Google Colab)
pip install pandas scikit-learn xgboost shap streamlit
📥 Step 2: Load Dataset
python
Copy
Edit
import pandas as pd

# Load dataset from URL or use Kaggle version
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/diabetes_readmission.csv')

df.head()
📌 Alternative Dataset (if needed):
Kaggle: Diabetes Readmission Dataset

🧹 Step 3: Clean the Data
python
Copy
Edit
# Drop irrelevant ID columns
df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

# Replace '?' with NaN and drop rows with too many missing values
df = df.replace('?', pd.NA)
df = df.dropna(thresh=40)

# Binary encode target: 1 = readmitted in <30 days, 0 = otherwise
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
🔄 Step 4: Encode Categorical Variables
python
Copy
Edit
# Drop high-cardinality/low-utility columns
drop_cols = ['weight', 'payer_code', 'medical_specialty']
df = df.drop(columns=drop_cols)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)
🧪 Step 5: Train/Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
🧠 Step 6: Train a Model (XGBoost)
python
Copy
Edit
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
📊 Step 7: Evaluate the Model
python
Copy
Edit
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
🔍 Step 8: Explain Predictions with SHAP
python
Copy
Edit
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)
🌐 Step 9: Deploy with Streamlit (Optional)
Create a file called app.py
python
Copy
Edit
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('readmission_model.pkl')

st.title("Patient Readmission Risk Prediction")

age = st.selectbox("Age", ['[60-70)', '[70-80)', '[50-60)'])
num_medications = st.slider("Number of medications", 0, 50, 10)

input_df = pd.DataFrame([[age, num_medications]], columns=['age', 'num_medications'])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("Prediction (1 = Will be readmitted):", prediction[0])
Run the app
bash
Copy
Edit
streamlit run app.py
💾 Step 10: Save the Model
python
Copy
Edit
import joblib

joblib.dump(model, 'readmission_model.pkl')
🚀 Bonus Improvements
Tune hyperparameters using GridSearchCV

Apply SMOTE to balance classes

Use cross-validation to boost evaluation reliability

Set up automatic retraining to address concept drift

Would you like this converted into a GitHub README.md or Colab Notebook? I can generate the file for you.




