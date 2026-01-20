import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Titanic Data
# We create 1000 passengers with realistic survival patterns
np.random.seed(42)
n_samples = 1000

# Features: Pclass (1-3), Sex (0=Male, 1=Female), Age (1-80), Fare (10-500)
data = {
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
    'Sex': np.random.choice([0, 1], n_samples), # 0: Male, 1: Female
    'Age': np.random.randint(1, 80, n_samples),
    'Fare': np.random.uniform(10, 500, n_samples)
}
df = pd.DataFrame(data)

# 2. Create Target (Survival Logic)
# Women (1) and First Class (1) had higher survival rates in history
# We add randomness to make it realistic
survival_score = (
    (df['Sex'] * 10) +           # Being female boosts score
    ((4 - df['Pclass']) * 5) +   # Higher class (lower number) boosts score
    (df['Fare'] / 50) +          # Higher fare boosts score
    (df['Age'] * -0.05) +        # Being younger boosts score slightly
    np.random.normal(0, 5, n_samples)
)

# Top 40% survive
threshold = np.percentile(survival_score, 60) 
df['Survived'] = (survival_score > threshold).astype(int)

# 3. Split Data
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save Model safely in the current folder
# (Since this script is inside /model/, it saves there directly)
current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, 'titanic_survival_model.pkl')

with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Success: Model trained and saved to {output_file}")