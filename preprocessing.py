from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Function to convert SMILES to fingerprint
def smiles_to_fingerprint(smiles, fp_size=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        gen = GetMorganGenerator(radius=2, fpSize=fp_size)
        fp = gen.GetFingerprint(mol)
        # Correct: create array with correct size
        arr = np.zeros((fp_size,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        # Return zero vector if invalid SMILES
        return np.zeros(fp_size, dtype=int)

# Load data
df = pd.read_csv('SMILES1_shuffled.csv')

# Generate fingerprints
fps1 = np.array([smiles_to_fingerprint(s) for s in df['Drug 1 SMILES']])
fps2 = np.array([smiles_to_fingerprint(s) for s in df['Drug 2 SMILES']])

# Combine fingerprints
X = np.hstack((fps1, fps2))  # Each sample will be 2048 features (1024 + 1024)

# Encode labels
y = df['Risk Level']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Low→0, Moderate→1, High→2

# Save the processed data
np.save('features.npy', X)
np.save('labels.npy', y_encoded)
print("✅ Preprocessing complete! Saved features.npy and labels.npy")
print("Classes:", label_encoder.classes_)
