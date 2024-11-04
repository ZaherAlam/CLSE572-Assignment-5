import argparse
import os
from sklearn.metrics import mean_squared_error
import numpy as np
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

# Argument Parsing for Hyperparameters
parser = argparse.ArgumentParser(description="Train a Morgan Fingerprint model")
parser.add_argument('--radius', type=int, default=2, help="Radius for Morgan Fingerprints")
parser.add_argument('--num_bits', type=int, default=2048, help="Number of bits for Morgan Fingerprints")
args = parser.parse_args()

with open('assignment5config.json', 'r') as f:
    config = json.load(f)
radius = config.get('radius', 2)
num_bits = config.get('num_bits', 2048)

# Load Dataset
data = pd.read_csv('data/lipophilicity.csv')
smiles = data['smiles']
y = data['exp']

# Generate Morgan Fingerprints
X = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius, num_bits) for s in smiles]

# Split data and Train model (for illustration)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions and RMSE
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Save Results
conda_env = os.getenv("CONDA_DEFAULT_ENV")
with open("results.txt", "w") as f:
    f.write(f"Test RMSE: {rmse}\n")
    f.write(f"Conda Environment: {conda_env}\n")
    f.write(f"Hyperparameters: radius={radius}, num_bits={num_bits}\n")
