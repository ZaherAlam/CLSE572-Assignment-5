import os
import sys
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to parse hyperparameters from command-line or config file
def parse_hyperparameters():
    if len(sys.argv) > 2:
        # Command-line arguments
        hidden_layer_sizes = tuple(map(int, sys.argv[1].strip('()').split(',')))
        max_iter = int(sys.argv[2])
    else:
        # Load hyperparameters from a JSON config file
        with open('assignment5.json', 'r') as file:
            config = json.load(file)
            hidden_layer_sizes = tuple(config.get('hidden_layer_sizes', (100,)))
            max_iter = config.get('max_iter', 1000)
    return hidden_layer_sizes, max_iter

# Function to compute Morgan Fingerprints
def compute_morgan_fingerprints(smiles_list):
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros((2048,)))  # Empty vector for invalid SMILES
    return np.array(fingerprints)

# Load dataset
df = pd.read_csv('lipophilicity.csv')

# Check for column names and strip any extra spaces
df.columns = df.columns.str.strip()

# Generate Morgan Fingerprints
morgan_fps = compute_morgan_fingerprints(df['smiles'])

# Split data into train and test sets
X_train_morgan, X_test_morgan, y_train, y_test = train_test_split(morgan_fps, df['exp'], test_size=0.2, random_state=42)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Parse hyperparameters
hidden_layer_sizes, max_iter = parse_hyperparameters()

# Initialize and train the MLPRegressor model
mlp_morgan = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)

# Scale the target
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train).ravel()  # Keep target as 1D for training
y_test_unscaled = y_test

# Fit the model
mlp_morgan.fit(X_train_morgan, y_train_scaled)

# Make predictions and rescale to original target scale
y_pred_morgan_scaled = mlp_morgan.predict(X_test_morgan)
y_pred_morgan_unscaled = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1))

# Evaluate performance using RMSE
rmse_morgan = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_morgan_unscaled))
print("RMSE for Morgan Fingerprints model:", rmse_morgan)

# Get current conda environment
conda_env = os.getenv("CONDA_DEFAULT_ENV")

# Save results to a text file
with open('model_results.txt', 'w') as result_file:
    result_file.write(f"RMSE: {rmse_morgan}\n")
    result_file.write(f"Conda Environment: {conda_env}\n")
    result_file.write(f"Hyperparameters: hidden_layer_sizes={hidden_layer_sizes}, max_iter={max_iter}\n")

# Print completion message
print("Model training complete. Results saved to 'model_results.txt'.")
