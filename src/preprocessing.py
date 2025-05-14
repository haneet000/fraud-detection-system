import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


df = pd.read_csv("C:/Users/devid/fraud-detection/data/creditcard.csv")
print(df.head())
print(df['Class'].value_counts())

# Separate X and y
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
