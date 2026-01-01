import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_excel("HR_Merged Dataset.xlsx")

# Keep only required columns
features = ["age", "jobrole", "monthlyincome", "overtime", "jobsatisfaction", "yearsatcompany"]
target = "attrition"
df = df[features + [target]]

# Create encoders and fit on FULL dataset (not just train split)
le_jobrole = LabelEncoder()
df["jobrole"] = le_jobrole.fit_transform(df["jobrole"])

le_overtime = LabelEncoder()
df["overtime"] = le_overtime.fit_transform(df["overtime"])

le_attrition = LabelEncoder()
df["attrition"] = le_attrition.fit_transform(df["attrition"])

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "attrition_model.pkl")
joblib.dump(le_jobrole, "le_jobrole.pkl")
joblib.dump(le_overtime, "le_overtime.pkl")
joblib.dump(le_attrition, "le_attrition.pkl")

print("✅ Model and encoders saved successfully in backend folder")
