import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/sample_employee_data.csv')

df['Education_Masters'] = (df['Education'] == 'Masters').astype(int)
df['Education_PhD'] = (df['Education'] == 'PhD').astype(int)
df['Skills_Python'] = df['Skills'].str.contains('Python').astype(int)
df['Skills_Excel'] = df['Skills'].str.contains('Excel').astype(int)
df['Skills_Java'] = df['Skills'].str.contains('Java').astype(int)
df['Skills_SQL'] = df['Skills'].str.contains('SQL').astype(int)

X = df[['Experience', 'Interview_Score', 'Education_Masters', 'Education_PhD',
        'Skills_Python', 'Skills_Excel', 'Skills_Java', 'Skills_SQL']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'models/salary_model.pkl')
print("✅ Model saved.")
