# ==========================
# Pirate Pain Prediction: LSTM / GRU / Transformer / CNN1D / MLP / GNN / Ensemble Models / TCN / TimesNet / BNN / CNN_RNN
# ==========================

import json
import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse
from scipy.stats import linregress
from scipy.fft import rfft, rfftfreq
import pickle

# --------------------------
# ARGUMENT PARSER
# --------------------------
parser = argparse.ArgumentParser(description="Pirate Pain Prediction")
parser.add_argument("--model_type", type=str, default="MLP",
                    help="Model type to use (LSTM / GRU / Transformer / CNN1D / MLP / Ensemble / GNN / TCN / TimesNet / BNN / CNN_RNN)")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--valid_split", type=float, default=0.2, help="Validation split ratio")
parser.add_argument("--early_stopping_patience", type=int, default=30, help="Early stopping patience")
parser.add_argument("--k_folds", type=int, default=7, help="Number of folds for cross-validation")
args = parser.parse_args()

# --------------------------
# CONFIG & SEEDS
# --------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# DIRECTORIES
# --------------------------
DATA_DIR = "dataset"
PRED_DIR = "prediction"
MODELS_DIR = "models"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
PLOTS_DIR = os.path.join(MODELS_DIR, timestamp, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(MODELS_DIR, timestamp, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------------------
# HYPERPARAMETERS
# --------------------------
MODEL_TYPE = args.model_type
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LR = args.learning_rate
VALID_SPLIT = args.valid_split
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
K_FOLDS = args.k_folds
PREPROCESSING_VISUAL_CHECK = False

PREPROCESSING_OPTS = {
    "gaussian": "none",   # "gaussian", "min", "min+log", "min+asinh", "boxcox", "none"
    "exp": "min+asinh",
    "pain": "none"
}

SCALER_OPTS = {
    "gaussian": "standard",   # "standard", "minmax", "robust", "none"
    "exp": "minmax",
    "pain": "none"
}

AUGMENTATION_OPTS = {
    "jitter": False,
    "jitter_sigma": 0.01,
    "scaling": False,
    "scaling_sigma": 0.05,
    "time_warp": False,
    "time_warp_max": 0.05,
    "random_mask": True,
    "mask_max_ratio": 0.05,
    "aug_prob": 0.3,
    "n_augmented": 1
}

LOAD_DATASET = False  
ADD_FEATURES = False
epsilon = 1e-6

print(f"Configuration: MODEL_TYPE={MODEL_TYPE}, BATCH_SIZE={BATCH_SIZE}, NUM_EPOCHS={NUM_EPOCHS}, LR={LR}, K_FOLDS={K_FOLDS}")


# --------------------------
# DATA LOADING
# --------------------------
X_train = pd.read_csv(os.path.join(DATA_DIR, "pirate_pain_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "pirate_pain_train_labels.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "pirate_pain_test.csv"))

categorical_cols = ['n_legs', 'n_hands', 'n_eyes']
X_train_enc = pd.get_dummies(X_train, columns=categorical_cols)
X_test_enc = pd.get_dummies(X_test, columns=categorical_cols)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Drop constant column
for c in ['joint_30']:
    if c in X_train_enc.columns:
        X_train_enc.drop(columns=[c], inplace=True)
        X_test_enc.drop(columns=[c], inplace=True)

# Divide based on feature distribution types
pain_cols = [f"pain_survey_{i}" for i in range(1, 5)]
gaussian_joints = [f"joint_{i:02d}" for i in range(0, 13)] + [f"joint_{i}" for i in range(26, 30)]
exp_joints = [f"joint_{i}" for i in range(13, 26)]
onehot_cols = [c for c in X_train_enc.columns if any(k in c for k in ['n_legs_', 'n_hands_', 'n_eyes_'])]
other_cols = ["time"]

# --------------------------
# PREPROCESSING
# --------------------------

def apply_preprocessing(X_train, X_test, cols, option):
    """Apply preprocessing per group."""
    X_train_mod, X_test_mod = X_train.copy(), X_test.copy()

    for col in cols:
        if col not in X_train_mod.columns:
            continue

        x_train, x_test = X_train_mod[col].values, X_test_mod[col].values
        min_val = min(x_train.min(), x_test.min())
        x_train_shift = x_train - min_val + epsilon
        x_test_shift = x_test - min_val + epsilon

        # --- apply chosen preprocessing ---
        if option == "gaussian":
            X_train_mod[col] = gaussian_filter1d(x_train, sigma=1)
            X_test_mod[col] = gaussian_filter1d(x_test, sigma=1)
            
        if option == "min":
            X_train_mod[col] = x_train_shift
            X_test_mod[col] = x_test_shift

        elif option == "min+log":
            X_train_mod[col] = np.log(x_train_shift)
            X_test_mod[col] = np.log(x_test_shift)

        elif option == "min+asinh":
            X_train_mod[col] = np.arcsinh(x_train_shift)
            X_test_mod[col] = np.arcsinh(x_test_shift)

        elif option == "boxcox":
            pt = PowerTransformer(method='box-cox', standardize=False)
            X_train_mod[col] = pt.fit_transform(x_train_shift.reshape(-1, 1)).ravel()
            X_test_mod[col] = pt.transform(x_test_shift.reshape(-1, 1)).ravel()

        elif option == "none":
            pass  # leave as is

        else:
            raise ValueError(f"Unknown preprocessing option: {option}")

    return X_train_mod, X_test_mod


def apply_scaler(X_train, X_test, cols, option):
    if len(cols) == 0 or option == "none":
        return X_train, X_test

    if option == "standard":
        scaler = StandardScaler()
    elif option == "minmax":
        scaler = MinMaxScaler()
    elif option == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {option}")

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_test


# lag analysis (autocorrelation)
from statsmodels.tsa.stattools import acf

def top3_acf_lags(df, feature_cols, sample_id=0, sample_idx_col='sample_index', time_col='time', max_lag=200):
    seq = df[df[sample_idx_col] == sample_id].sort_values(time_col)
    results = []

    for feature in feature_cols:
        y = seq[feature].values
        nlags = min(max_lag, len(y)-2)
        acf_vals = acf(y, fft=False, nlags=nlags)

        top3_idx = np.argsort(acf_vals[1:])[::-1][:3] + 1
        top3_vals = acf_vals[top3_idx]

        for lag, val in zip(top3_idx, top3_vals):
            results.append({
                "feature": feature,
                "lag": lag,
                "acf_value": val
            })

    df_top3 = pd.DataFrame(results)
    return df_top3

if PREPROCESSING_VISUAL_CHECK:
    top3_acf_table = top3_acf_lags(X_train_enc, gaussian_joints + exp_joints, sample_id=0)
    for feature, group in top3_acf_table.groupby('feature'):
        print(f"\nFeature: {feature}")
        top3 = group.sort_values('acf_value', ascending=False)
        for i, row in top3.iterrows():
            print(f"  Lag: {row['lag']}, ACF: {row['acf_value']:.4f}")

# --------------------------
# PIPELINE
# --------------------------

# Gaussian
X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, gaussian_joints, PREPROCESSING_OPTS["gaussian"])
X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, gaussian_joints, SCALER_OPTS["gaussian"])

# Exponential
X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, exp_joints, PREPROCESSING_OPTS["exp"])
X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, exp_joints, SCALER_OPTS["exp"])

# Pain survey integers
X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, pain_cols, PREPROCESSING_OPTS["pain"])
X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, pain_cols, SCALER_OPTS["pain"])

# Final feature columns
feature_cols = pain_cols + gaussian_joints + exp_joints + onehot_cols + other_cols


label_mapping = {label: idx for idx, label in enumerate(sorted(y_train['label'].unique()))}
y_train['label_encoded'] = y_train['label'].map(label_mapping)


# --------------------------
# FUNZIONE FEATURE ENGINEERING
# --------------------------
def add_features(df, joint_cols, time_col='time', window_sizes=[], time_mods=[14,24,40]):
    """
    Aggiunge features dai joint e dal tempo (moduli).
    """
    print(f"Adding BASE engineered features for {len(joint_cols)} joint columns...")
    df_new = df.copy()
    new_features = {}

    # --- Features dai joint ---
    for col in tqdm(joint_cols, desc="Base Feature Engineering", ncols=80):
        x = df_new[col].fillna(0).values

        # Derivative 1° ordine
        # new_features[f"{col}_diff1"] = np.diff(np.insert(x, 0, x[0]))

        # Rolling statistics
        for w in window_sizes:
            roll_mean = uniform_filter1d(x, size=w, mode='nearest')
            cumsum = uniform_filter1d(x**2, size=w, mode='nearest')
            roll_std = np.sqrt(np.maximum(cumsum - roll_mean**2, 0))

            new_features[f"{col}_rollmean_{w}"] = roll_mean
            new_features[f"{col}_rollstd_{w}"] = roll_std

    # --- Features cicliche del tempo ---
    if time_col in df_new.columns:
        t = df_new[time_col].fillna(0).values
        for mod in time_mods:
            new_features[f"time_mod{mod}"] = t % mod

    # Merge nuove features
    new_df = pd.DataFrame(new_features, index=df_new.index)

    # Standardizzazione
    scaler = StandardScaler()
    new_df[new_df.columns] = scaler.fit_transform(new_df)

    df_new = pd.concat([df_new, new_df], axis=1)
    return df_new


# --------------------------
# ESEMPIO USO
# --------------------------
all_joint_cols = gaussian_joints + exp_joints
time_mods = [14, 24]

print("\n--- Feature Engineering Pipeline ---")
os.makedirs("dataset", exist_ok=True)

# --- INFO DATASET ---
'''
print("\n--- Dataset info ---")
print(f"X_train shape: {X_train_enc.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test_enc.shape}")
''' 

if PREPROCESSING_VISUAL_CHECK:
    num_train_samples = X_train_enc['sample_index'].nunique()
    num_test_samples = X_test_enc['sample_index'].nunique()
    print(f"Number of training sequences: {num_train_samples}")
    print(f"Number of test sequences: {num_test_samples}")
    num_features = len(feature_cols)
    print(f"Number of features used: {num_features}")
    timesteps_per_seq = X_train_enc.groupby('sample_index').size()
    print(f"Timesteps per training sequence: min={timesteps_per_seq.min()}, max={timesteps_per_seq.max()}, mean={int(timesteps_per_seq.mean())}")
    timesteps_per_seq_test = X_test_enc.groupby('sample_index').size()
    print(f"Timesteps per test sequence: min={timesteps_per_seq_test.min()}, max={timesteps_per_seq_test.max()}, mean={int(timesteps_per_seq_test.mean())}")
    label_counts = y_train['label'].value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    print("\nLabel to index mapping:")
    print(label_mapping)

if LOAD_DATASET:
    with open("dataset/X_train_enc.pkl", "rb") as f:
        X_train_enc = pickle.load(f)
    with open("dataset/X_test_enc.pkl", "rb") as f:
        X_test_enc = pickle.load(f)
    print("Dataset caricati da 'dataset/'. Nessuna nuova feature aggiunta.")
    feature_cols = [c for c in X_train_enc.columns if c not in categorical_cols]

elif ADD_FEATURES:
    X_train_enc = add_features(X_train_enc, all_joint_cols, time_col='time', time_mods=time_mods)
    X_test_enc = add_features(X_test_enc, all_joint_cols, time_col='time', time_mods=time_mods)

    with open("dataset/X_train_enc.pkl", "wb") as f:
        pickle.dump(X_train_enc, f)
    with open("dataset/X_test_enc.pkl", "wb") as f:
        pickle.dump(X_test_enc, f)

    print("New features added and datasets saved to 'dataset/'.")
    feature_cols = [c for c in X_train_enc.columns if c not in categorical_cols + ['time', 'sample_index']]

else:
    print("Using raw preprocessed dataset (no feature engineering).")
    feature_cols = pain_cols + gaussian_joints + exp_joints + onehot_cols

print(f"Total feature columns: {len(feature_cols)}")


# --------------------------
# Data check
# --------------------------
print("Final preprocessing summary:")
print(f"Pain cols: {pain_cols}")
print(f"Gaussian joints: {gaussian_joints}")
print(f"Exponential joints: {exp_joints}")
print(f"One-hot cols: {onehot_cols}")
print(f"Preprocessing: {PREPROCESSING_OPTS}")
print(f"Scalers: {SCALER_OPTS}")

if PREPROCESSING_VISUAL_CHECK:
    for group_name, cols in {
        "Gaussian": gaussian_joints[:3],
        "Exponential": exp_joints[:3],
        "Pain": pain_cols[:3]
    }.items():
        for col in cols:
            plt.figure(figsize=(6, 3))
            plt.hist(X_train_enc[col], bins=50, alpha=0.7)
            plt.title(f"{group_name} → {col}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

print(f"Final feature columns ({len(feature_cols)}): {feature_cols[:10]} ...")


def analyze_features_for_group(df, feature_cols, group_ids, time_col='time', sample_idx_col='sample_index', save_plots=False, plot_dir="plots"):

    if isinstance(group_ids, (int, str)):
        group_ids = [group_ids]

    df_group = df[df[sample_idx_col].isin(group_ids)]

    if df_group.empty:
        print(f"Nessun dato trovato per i gruppi {group_ids}")
        return {}

    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    stats_summary = {}

    for col in feature_cols:
        print(f"\n--- Analysis for {col} (groups {group_ids}) ---")

        series = df_group[col]

        is_bool = set(series.unique()).issubset({0,1})

        stats = {}
        if is_bool:

            counts_0 = (series == 0).sum()
            counts_1 = (series == 1).sum()
            total = counts_0 + counts_1
            prop_0 = counts_0 / total if total > 0 else np.nan
            prop_1 = counts_1 / total if total > 0 else np.nan
            stats = {
                'counts_0': counts_0,
                'counts_1': counts_1,
                'prop_0': prop_0,
                'prop_1': prop_1
            }
        else:
            stats = {
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                '25%': series.quantile(0.25),
                '75%': series.quantile(0.75),
            }

        stats_summary[col] = stats
        for k, v in stats.items():
            print(f"{k}: {v}")


        plt.figure(figsize=(8,4))
        for idx, df_sub in df_group.groupby(sample_idx_col):
            plt.plot(df_sub[time_col], df_sub[col], label=f"{sample_idx_col}={idx}", alpha=0.8)
        plt.title(f"{col} over time for group(s) {group_ids}")
        plt.xlabel(time_col)
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{col}_groups_{'_'.join(map(str, group_ids))}.png"))
        plt.show()


        plt.figure(figsize=(6,4))
        if is_bool:
            sns.countplot(x=col, data=df_group)
        else:
            sns.histplot(series, bins=30, kde=True)
        plt.title(f"Distribution of {col} (group {group_ids})")
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{col}_dist_groups_{'_'.join(map(str, group_ids))}.png"))
        plt.show()

    return stats_summary


if PREPROCESSING_VISUAL_CHECK:
    stats_summary = analyze_features_for_group(X_train_enc, feature_cols, group_ids=4)
    
    print("X_train_enc shape:", X_train_enc.shape) 

    print("y_train label column:")
    print(y_train['label'].head())
    print("y_train encoded labels:")
    print(y_train['label_encoded'].unique())

    sns.countplot(x='label', data=y_train)
    plt.title("Distribution of original labels")
    plt.grid(True)
    plt.show()

    sns.countplot(x='label_encoded', data=y_train)
    plt.title("Distribution of encoded labels")
    plt.grid(True)
    plt.show()



# --------------------------
# AUGMENTATION FUNCTIONS
# --------------------------
def jitter(x, sigma):
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)

def scaling(x, sigma):
    factor = np.random.normal(1.0, sigma, (x.shape[1],))
    return x * factor.astype(np.float32)

def time_warp(x, max_warp):
    from scipy.interpolate import interp1d
    L, D = x.shape
    warp = np.linspace(0, L-1, num=L) + np.random.uniform(-max_warp*L, max_warp*L, size=L)
    warp = np.clip(warp, 0, L-1)
    f = interp1d(np.arange(L), x, axis=0, kind='linear', fill_value="extrapolate")
    return f(warp).astype(np.float32)

def random_mask(x, max_ratio):
    L, D = x.shape
    mask_len = int(L * np.random.uniform(0, max_ratio))
    if mask_len == 0: return x
    start = np.random.randint(0, L - mask_len)
    x[start:start+mask_len, :] = 0
    return x

def augment_sequence(x, opts=AUGMENTATION_OPTS):
    x_aug = x.copy()
    if opts["jitter"]: x_aug = jitter(x_aug, opts["jitter_sigma"])
    if opts["scaling"]: x_aug = scaling(x_aug, opts["scaling_sigma"])
    if opts["time_warp"]: x_aug = time_warp(x_aug, opts["time_warp_max"])
    if opts["random_mask"]: x_aug = random_mask(x_aug, opts["mask_max_ratio"])
    return x_aug

# --------------------------
# SEQUENCE DATASET WITH AUGMENTATION
# --------------------------
class PiratePainSeqDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X, self.y = [], []
        self.augment = augment

        if y is not None:
            y_map = y.set_index('sample_index')['label'].map(label_mapping)

        for idx, df_sub in X.groupby('sample_index'):
            seq = df_sub[feature_cols].values.astype(np.float32)
            self.X.append(seq)
            if y is not None:
                if idx not in y_map.index:
                    raise ValueError(f"sample_index {idx} non trovato in y_train")
                self.y.append(y_map.loc[idx])

            if self.augment and y is not None:
                for _ in range(AUGMENTATION_OPTS["n_augmented"]):
                    self.X.append(augment_sequence(seq))
                    self.y.append(y_map.loc[idx])

        if y is not None:
            self.y = np.array(self.y)
        else:
            self.y = None

        # print(f"\nTotal sequences loaded: {len(self.X)}")
        # if y is not None:
            #print(f"Total labels loaded: {len(self.y)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            return x, torch.tensor(self.y[idx], dtype=torch.long)
        return x


def collate_fn(batch):
    x, y = zip(*batch) if isinstance(batch[0], tuple) else (batch, None)
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
    if y is not None:
        y = torch.stack(y)
        return x_padded, y
    return x_padded

# --------------------------
# DATASETS & LOADERS (GroupKFold) 
# --------------------------
print("\n--- Creating GroupKFold datasets & loaders ---")

# Array delle sequenze
unique_samples = y_train['sample_index'].values          
y_seq = y_train['label'].map(label_mapping).values       
groups = unique_samples                                    
X_dummy = np.zeros((len(unique_samples), 1))

gkf = GroupKFold(n_splits=K_FOLDS)
folds = list(gkf.split(X_dummy, y_seq, groups=groups))

print(f"\nCreated {K_FOLDS} folds for GroupKFold cross-validation.")

test_dataset = PiratePainSeqDataset(X_test_enc, augment=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for i, (train_idx, val_idx) in enumerate(folds, 1):
    train_samples = unique_samples[train_idx]
    val_samples = unique_samples[val_idx]
    # print(f"Fold {i}: {len(train_samples)} train samples, {len(val_samples)} val samples")


'''
# --------------------------
# SEQUENCE DATASET
# --------------------------
class PiratePainSeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = [], []

        if y is not None:
            y_map = y.set_index('sample_index')['label'].map(label_mapping)
            # print("\nMapping sample_index -> label:\n", y_map.head())

        for idx, df_sub in X.groupby('sample_index'):
            # print(f"\nProcessing sample_index {idx}, seq length {len(df_sub)}")
            seq = df_sub[feature_cols].values.astype(np.float32)
            self.X.append(seq)
            if y is not None:
                if idx not in y_map.index:
                    raise ValueError(f"sample_index {idx} non trovato in y_train")
                self.y.append(y_map.loc[idx])

        print(f"\nTotal sequences loaded: {len(self.X)}")
        if y is not None:
            self.y = np.array(self.y)
            print(f"Total labels loaded: {len(self.y)}")
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            return x, torch.tensor(self.y[idx], dtype=torch.long)
        return x


# Collate function padding sequenze
def collate_fn(batch):
    x, y = zip(*batch) if isinstance(batch[0], tuple) else (batch, None)
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
    if y is not None:
        y = torch.stack(y)
        return x_padded, y
    return x_padded

# Datasets & Loaders
print("\n--- Creating datasets & loaders ---")
dataset = PiratePainSeqDataset(X_train_enc, y_train)
train_idx, val_idx = train_test_split(
    np.arange(len(dataset)), test_size=VALID_SPLIT, stratify=dataset.y, random_state=SEED
)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

test_dataset = PiratePainSeqDataset(X_test_enc)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

print(f"\nTrain sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}, Test sequences: {len(test_dataset)}")
'''


# --------------------------
# MODELS
# --------------------------
input_dim = len(feature_cols)
num_classes = len(label_mapping)

# ATTENTIVE POOLING 1D
class AttentivePooling1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        x_pooled = (x * weights).sum(dim=1)
        return x_pooled

# LSTM MODEL 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
    
# GRU 
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size=512, num_layers=4, num_classes=10, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.att_pool = AttentivePooling1D(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        out = self.att_pool(out)
        return self.fc(out)
    

# TIMESNET MODEL
class AttentivePooling1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, L, H]
        weights = torch.softmax(self.att(x), dim=1)  # [B,L,1]
        return (x * weights).sum(dim=1)             # [B,H]


class TimesBlock(nn.Module):
    def __init__(self, hidden_dim, periods=[4,8,16,32], dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.periods = periods
        self.convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1,p), padding=(0,p//2))
            for p in periods
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,L,H]
        B, L, H = x.size()
        x_in = x
        conv_outs = []

        for i, p in enumerate(self.periods):
            if L < p:
                continue
            # reshape for Conv2d: [B,H,L,1]
            x_reshaped = x.permute(0,2,1).unsqueeze(-1)
            conv_out = F.relu(self.convs[i](x_reshaped))   # [B,H,L,1]
            # safely squeeze last dim
            conv_out = conv_out[..., 0].permute(0,2,1)    # [B,L,H]
            conv_outs.append(conv_out)

        if len(conv_outs) == 0:
            x_out = x
        else:
            x_out = sum(conv_outs) / len(conv_outs)

        x_out = self.dropout(x_out)
        return self.layer_norm(x_in + x_out)  


class TimesNet(nn.Module):
    """Full TimesNet model for classification/regression"""
    def __init__(self, input_dim, num_classes, hidden_dim=192, num_blocks=4,
                 max_seq_len=500, dropout=0.2, periods=[4,8,16,32]):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        self.blocks = nn.ModuleList([TimesBlock(hidden_dim, periods=periods, dropout=dropout)
                                     for _ in range(num_blocks)])
        self.att_pool = AttentivePooling1D(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,L,input_dim]
        B, L, _ = x.size()
        x = self.input_fc(x)
        x = x + self.pos_enc[:, :L, :]  
        for block in self.blocks:
            x = block(x)
        x = self.att_pool(x)
        x = self.dropout(x)
        return self.fc(x)


# TRANSFORMER 
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=192, nhead=4, num_layers=2, max_seq_len=500, dropout=0.4):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim*4,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.att_pool = AttentivePooling1D(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.input_fc(x)
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        x = self.att_pool(x)
        x = self.dropout(x)
        return self.fc(x)

''' 
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, nhead, num_layers, max_seq_len, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim*4  # feed-forward expansion
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.input_fc(x)
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
'''

# TCN MODEL
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y

class TemporalBlockInception(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dilation=1, dropout=0.3):
        super().__init__()
        # Multi-kernel convolutions
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = ((k-1)*dilation)//2
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad, dilation=dilation)
            )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock1D(out_channels)
        # Residual connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = sum(conv(x) for conv in self.convs)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.se(out)
        out += residual
        return self.relu(out)

class TCNModel(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels, kernel_sizes_list, dilation_list=None, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = input_dim

        if dilation_list is None:
            dilation_list = [2**i for i in range(len(conv_channels))]

        for out_channels, ks, d in zip(conv_channels, kernel_sizes_list, dilation_list):
            if isinstance(ks, int):
                ks = [ks]
            layers.append(
                TemporalBlockInception(in_channels, out_channels, kernel_sizes=ks, dilation=d, dropout=dropout)
            )
            in_channels = out_channels

        layers.append(nn.AdaptiveMaxPool1d(1))  # global aggregation
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1,2)  # [batch, features, seq_len]
        x = self.tcn(x)
        x = x.squeeze(-1)     # [batch, channels]
        return self.fc(x)


# CNN1D MODEL
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y

class ResidualBlockInception1DDilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dilation=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k//2)*dilation, dilation=dilation)
            )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock1D(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = sum(conv(x) for conv in self.convs)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels, kernel_sizes_list, dilation_list=None, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = input_dim
        if dilation_list is None:
            dilation_list = [1]*len(conv_channels)  # default dilation=1
        for out_channels, ks, d in zip(conv_channels, kernel_sizes_list, dilation_list):
            if isinstance(ks, int):
                ks = [ks]
            layers.append(
                ResidualBlockInception1DDilated(in_channels, out_channels, kernel_sizes=ks, dilation=d, dropout=dropout)
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1,2)  # [batch, features, seq_len]
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)
    
'''
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        y = x.mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y

class ResidualBlockInception1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
            )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock1D(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = sum(conv(x) for conv in self.convs)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels, kernel_sizes_list, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = input_dim
        for out_channels, ks in zip(conv_channels, kernel_sizes_list):
            if isinstance(ks, int):
                ks = [ks]
            layers.append(ResidualBlockInception1D(in_channels, out_channels, kernel_sizes=ks, dropout=dropout))
            in_channels = out_channels
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1,2) 
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)
'''

# --------------------------
# MLP MODEL
# --------------------------


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.mean(dim=1)  # aggregazione temporale (puoi cambiare con sum/max o LSTM)
        return self.net(x)
    
'''
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        self.att_pool = AttentivePooling1D(input_dim)
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.att_pool(x)
        return self.net(x)
''' 
    
# --------------------------
# GNN MODEL
# --------------------------
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout, concat=True))

        self.relu = nn.ReLU()
        self.att_pool = AttentivePooling1D(hidden_dim*heads)
        self.fc = nn.Linear(hidden_dim*heads, num_classes)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """
        batch_size, seq_len, feat_dim = x.size()

        device = x.device
        edge_indices = []
        for b in range(batch_size):
            row = torch.arange(seq_len-1, device=device) + b*seq_len
            col = torch.arange(1, seq_len, device=device) + b*seq_len
            edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
            edge_indices.append(edge_index)
        edge_index = torch.cat(edge_indices, dim=1)  

        x = x.reshape(batch_size*seq_len, feat_dim)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, seq_len, -1)
        x = self.att_pool(x)

        return self.fc(x)

'''
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # --- GCN layers ---
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """
        batch_size, seq_len, feat_dim = x.size()
        device = x.device

        edge_indices = []
        for b in range(batch_size):
            row = torch.arange(seq_len-1, device=device) + b*seq_len
            col = torch.arange(1, seq_len, device=device) + b*seq_len
            edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
            edge_indices.append(edge_index)
        edge_index = torch.cat(edge_indices, dim=1)  # concateni tutti i batch

        # Flatten batch: [batch*seq_len, input_dim]
        x = x.view(batch_size*seq_len, feat_dim)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout_layer(x)

        x = x.view(batch_size, seq_len, -1)  # [batch, seq_len, hidden_dim]
        x = x.mean(dim=1)  # aggrega timestep

        return self.fc(x)
'''

# BAYESIAN NEURAL NETWORK (BNN) MODEL
import torchbnn as bnn

class BayesianModel(nn.Module):
    def __init__(self, input_dim, num_classes,
                 hidden_sizes=[128,64],
                 dropout=0.2,
                 prior_mu=0.0,
                 base_prior_sigma=0.05,
                 kl_annealing_epochs=20):
        super().__init__()
        self.kl_annealing_epochs = kl_annealing_epochs
        self.current_epoch = 0

        # Encoder sequenza (deterministico)
        self.gru = nn.GRU(input_dim, hidden_sizes[0], batch_first=True, bidirectional=True)
        feat_dim = hidden_sizes[0]*2

        # Attentive pooling
        self.attn = nn.Linear(feat_dim, 1)

        # Bayesian classifier
        layers = []
        in_dim = feat_dim
        for idx, h in enumerate(hidden_sizes[1:]):
            sigma = base_prior_sigma * (1 + idx*0.5)
            layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=sigma, in_features=in_dim, out_features=h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        # Output bayesiano
        layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=base_prior_sigma, in_features=in_dim, out_features=num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, time, features]
        out, _ = self.gru(x)
        weights = torch.softmax(self.attn(out), dim=1)
        x_pooled = (out * weights).sum(dim=1)
        return self.net(x_pooled)

    def kl_loss(self):
        kl = 0
        kl_fn = bnn.BKLLoss(reduction="sum")
        for module in self.modules():
            if isinstance(module, bnn.BayesLinear):
                kl += kl_fn(module)
        # Annealing KL: gradualmente aumentiamo il peso
        kl_weight = min(1.0, self.current_epoch / self.kl_annealing_epochs)
        return kl * kl_weight


# CNN + GRU 
class CNN_RNN(nn.Module):
    def __init__(self, input_dim, num_classes,
                 cnn_out_channels=64,
                 cnn_kernel=5,
                 gru_hidden=128,
                 gru_layers=1,
                 dropout=0.3):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.gru = None  

        self.layer_norm = nn.LayerNorm(gru_hidden*2)
        self.att_pool = nn.Linear(gru_hidden*2, 1)
        self.fc = nn.Linear(gru_hidden*2, num_classes)
        self.dropout_gru = nn.Dropout(dropout)

    def forward(self, x, seq_features=None):
        
        B, T, F = x.size()

        x_cnn = x.transpose(1,2)         # [B, F, T]
        x_cnn = self.cnn(x_cnn)          # [B, C, T]
        x_cnn = self.relu(x_cnn)
        x_cnn = self.dropout(x_cnn)
        x_cnn = x_cnn.transpose(1,2)     # [B, T, C]
        gru_input = torch.cat([x, x_cnn], dim=-1)  # [B, T, F + C]

        if seq_features is not None:
            seq_tiled = seq_features.unsqueeze(1).repeat(1, T, 1)
            gru_input = torch.cat([gru_input, seq_tiled], dim=-1)  # [B, T, F + C + extra]

        if self.gru is None:
            input_dim_gru = gru_input.size(-1)
            self.gru = nn.GRU(
                input_dim_gru, self.gru_hidden, num_layers=self.gru_layers,
                batch_first=True, bidirectional=True,
                dropout=0. if self.gru_layers==1 else self.dropout_gru.p
            ).to(gru_input.device)

        out, _ = self.gru(gru_input)           # [B, T, H*2]
        out = self.layer_norm(out)

        att_weights = torch.softmax(self.att_pool(out), dim=1)  # [B, T, 1]
        out = (out * att_weights).sum(dim=1)                   # [B, H*2]
        out = self.dropout_gru(out)

        return self.fc(out)


# --------------------------
# ENSEMBLE MODEL
# --------------------------
class EnsembleModel(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.device = device
        # --- CNN1D (Inception + Dilated + SE) ---
        self.cnn = CNN1DModel(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=[48, 96, 160],
            kernel_sizes_list=[
                [3, 5, 7],
                [5, 7, 9],
                [7, 9, 11]
            ],
            dropout=0.4
        )
        # --- GRU ---
        self.gru = GRUModel(
            input_dim=input_dim,
            hidden_size=192,
            num_layers=3,
            num_classes=num_classes,
            dropout=0.4
        )
        # --- LSTM ---
        self.lstm = LSTMModel(
            input_dim=input_dim,
            hidden_size=192,
            num_layers=3,
            num_classes=num_classes,
            dropout=0.4
        )
        # --- Transformer ---
        self.transformer = TransformerModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=96,
            nhead=2,
            num_layers=3,
            max_seq_len=500,
            dropout=0.3
        )
        # --- MLP ---
        self.mlp = MLPModel(
            input_dim=input_dim,
            hidden_sizes=[384, 192, 128],
            num_classes=num_classes,
            dropout=0.4
        )
        # --- Trainable ensemble weights ---
        self.model_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        """
        Weighted ensemble forward pass
        """
        out_cnn = self.cnn(x)
        out_gru = self.gru(x)
        out_lstm = self.lstm(x)
        out_trans = self.transformer(x)
        out_mlp = self.mlp(x)

        logits = torch.stack([out_cnn, out_gru, out_lstm, out_trans, out_mlp], dim=1)
        weights = self.softmax(self.model_weights).unsqueeze(0).unsqueeze(-1)
        weighted_logits = (logits * weights).sum(dim=1)
        return weighted_logits

    def predict(self, x):
        """
        Soft ensemble prediction
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(F.softmax(logits, dim=-1), dim=-1)

    def predict_majority(self, x, confidence_threshold=0.5):
        """
        Hard voting with fallback to weighted ensemble
        """
        with torch.no_grad():
            preds = [
                torch.argmax(F.softmax(self.cnn(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.gru(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.lstm(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.transformer(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.mlp(x), dim=-1), dim=-1)
            ]
            preds = torch.stack(preds, dim=1)
            
            majority, counts = torch.mode(preds, dim=1)
            confident = (counts.float() / preds.size(1)) >= confidence_threshold
            fallback_mask = ~confident

            if fallback_mask.any():
                weighted_preds = self.predict(x[fallback_mask])
                majority[fallback_mask] = weighted_preds

            return majority


# --------------------------
# Select model architecture
# --------------------------
models = []

for fold in range(K_FOLDS):

    if MODEL_TYPE=="LSTM":
        lstm_hidden_size = 128  
        lstm_num_layers = 2        
        lstm_dropout = 0.3
        model = LSTMModel(input_dim, lstm_hidden_size, lstm_num_layers, num_classes, lstm_dropout).to(device)

    elif MODEL_TYPE=="GRU":
        gru_hidden_size = 128
        gru_num_layers = 2
        gru_dropout = 0.3
        model = GRUModel(input_dim, gru_hidden_size, gru_num_layers, num_classes, gru_dropout).to(device)

    elif MODEL_TYPE=="Transformer":
        transformer_hidden_dim = 96 
        transformer_nhead = 2
        transformer_num_layers = 3  
        transformer_max_seq_len = 500
        transformer_dropout = 0.3
        model = TransformerModel(
            input_dim, num_classes, transformer_hidden_dim, transformer_nhead,
            transformer_num_layers, transformer_max_seq_len, transformer_dropout
        ).to(device)

    elif MODEL_TYPE=="TimesNet":
        model = TimesNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=128,
            num_blocks=3,
            max_seq_len=500,
            dropout=0.3
        ).to(device)

    elif MODEL_TYPE=="CNN1D":
        cnn_conv_channels = [48, 96, 160]
        cnn_kernel_sizes = [
            [3, 5, 7],
            [5, 7, 9], 
            [7, 9, 11]
        ]
        cnn_dropout = 0.4

        model = CNN1DModel(
            input_dim, 
            num_classes, 
            cnn_conv_channels, 
            cnn_kernel_sizes, 
            dropout=cnn_dropout
        ).to(device)
        
    elif MODEL_TYPE=="CNN_RNN":
        cnn_out_channels = 64       # output channels del CNN base
        cnn_kernel = 5              # kernel size semplice
        cnn_dropout = 0.3

        gru_hidden = 192            # hidden size GRU
        gru_layers = 2              # numero layer GRU
        gru_dropout = 0.3           # dropout GRU

        model = CNN_RNN(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel=cnn_kernel,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=cnn_dropout
        ).to(device)

        
        
    elif MODEL_TYPE=="TCN":
        tcn_conv_channels = [48, 96, 160]  
        tcn_kernel_sizes = [
            [3, 5, 7],
            [5, 7, 9],
            [7, 9, 11]
        ]
        tcn_dilation_list = [1, 2, 4]  
        tcn_dropout = 0.4

        model = TCNModel(
            input_dim,
            num_classes,
            tcn_conv_channels,
            tcn_kernel_sizes,
            dilation_list=tcn_dilation_list,
            dropout=tcn_dropout
        ).to(device)

    elif MODEL_TYPE=="MLP":
        mlp_hidden_sizes = [256, 128, 64]
        mlp_dropout = 0.3
        model = MLPModel(input_dim, mlp_hidden_sizes, num_classes, mlp_dropout).to(device)

    elif MODEL_TYPE=="GNN":
        gnn_hidden_dim = 96
        gnn_num_layers = 2
        gnn_dropout = 0.2
        model = GNNModel(
            input_dim=input_dim,
            hidden_dim=gnn_hidden_dim,
            num_classes=num_classes,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout
        ).to(device)
        
    elif MODEL_TYPE=="BNN":
        bnn_hidden_sizes = [256, 128, 64]
        bnn_dropout = 0.3
        model = BayesianModel(
            input_dim=input_dim,
            hidden_sizes=bnn_hidden_sizes,
            num_classes=num_classes,
            dropout=bnn_dropout
        ).to(device)

    elif MODEL_TYPE=="Ensemble":
        model = EnsembleModel(input_dim, num_classes, device).to(device)

    else:
        raise ValueError("MODEL_TYPE must be one of: LSTM, GRU, Transformer, CNN1D, MLP, GNN, Ensemble")

    # print(f"\n Model {MODEL_TYPE} initialized for fold {fold+1}/{K_FOLDS}")

    models.append(model)

# --------------------------
# LOSS & OPTIMIZER
# --------------------------
'''
classes = np.unique(y_train['label'].map(label_mapping))
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train['label'].map(label_mapping))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class CombinedLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha 
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        ce_loss_none = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(input, target)
        pt = torch.exp(-ce_loss_none)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss_none).mean()
        combined = self.alpha * ce_loss + (1 - self.alpha) * focal_loss
        return combined

# criterion = CombinedLoss(gamma=2, weight=class_weights, alpha=1.0)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)


if MODEL_TYPE in []:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=5,
        min_lr=1e-6
    )
elif MODEL_TYPE in ["Transformer", "TimesNet", "CNN1D", "MLP", "GNN", "TCN", "LSTM", "GRU", "CNN_RNN"]:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2, 
        eta_min=1e-5
    )
elif MODEL_TYPE == "BNN":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=10,
        min_lr=1e-6
    )
elif MODEL_TYPE == "Ensemble":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )


else:
    raise ValueError("MODEL_TYPE must be one of: LSTM, GRU, Transformer, CNN1D, MLP, GNN, Ensemble, TCN, TimesNet")

print(f"\n Loss & Optimizer set for {MODEL_TYPE}")

'''

# --------------------------
# GROUPKFOLD TRAINING LOOP
# --------------------------
print("\n--- GroupKFold Training ---")

# Sequenze uniche e labels
unique_samples = y_train['sample_index'].values         
y_seq = y_train['label'].map(label_mapping).values   
groups = unique_samples                                 

# Dummy X per GroupKFold
X_dummy = np.zeros((len(unique_samples), 1))
gkf = GroupKFold(n_splits=K_FOLDS)

fold_scores = []
all_logs = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_seq, groups=groups), 1):
    # print(f"\n--- Fold {fold}/{K_FOLDS} ---")

    train_samples = unique_samples[train_idx]
    val_samples = unique_samples[val_idx]

    train_dataset = PiratePainSeqDataset(
        X_train_enc[X_train_enc['sample_index'].isin(train_samples)],
        y_train[y_train['sample_index'].isin(train_samples)],
        augment=True
    )
    val_dataset = PiratePainSeqDataset(
        X_train_enc[X_train_enc['sample_index'].isin(val_samples)],
        y_train[y_train['sample_index'].isin(val_samples)],
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # print(f"Train sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}")


# --------------------------
# TRAINING LOOP PER FOLD
# --------------------------
fold_scores = []
all_logs = []

hyperparams = {
    "MODEL_TYPE": MODEL_TYPE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LR,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "K_FOLDS": K_FOLDS,
    "WEIGHT_DECAY": 1e-4,
    "MAX_GRAD_NORM": 5.0,
    
    "PREPROCESSING_OPTS": PREPROCESSING_OPTS,
    "SCALER_OPTS": SCALER_OPTS,
    "AUGMENTATION_OPTS": AUGMENTATION_OPTS,

    "SEED": SEED,
    "DEVICE": str(device),
    "TIMESTAMP": timestamp
}

hyperparams_path = os.path.join(PRED_DIR, f"hyperparameters_{MODEL_TYPE}_{timestamp}.json")
with open(hyperparams_path, "w") as f:
    json.dump(hyperparams, f, indent=4)

print(f"Hyperparameters saved to {hyperparams_path}")


log_txt_path = os.path.join(PRED_DIR, f"training_log_{MODEL_TYPE}.txt")
with open(log_txt_path, "w") as log_file:

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_seq, groups=groups), 1):
        print(f"\n--- Fold {fold}/{K_FOLDS} ---")
        log_file.write(f"\n--- Fold {fold}/{K_FOLDS} ---\n")

        model = models[fold-1]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if MODEL_TYPE in ["Transformer", "TimesNet", "CNN1D", "MLP", "GNN", "TCN", "LSTM", "GRU", "CNN_RNN"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-5
            )
        elif MODEL_TYPE == "BNN":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=10, min_lr=1e-6
            )
        elif MODEL_TYPE == "Ensemble":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
            )

        best_val_f1 = float('-inf')
        early_stop_counter = 0
        logs = []

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            total_loss = 0

            for X_batch, y_batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} Training"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)

                if MODEL_TYPE == "BNN":
                    ce_loss = criterion(y_pred, y_batch)
                    kl = model.kl_loss() / len(train_loader.dataset)
                    loss = ce_loss + 1e-4 * kl
                else:
                    loss = criterion(y_pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
                    all_preds.append(y_pred.argmax(1).cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            val_acc = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            logs.append({
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'lr': current_lr
            })

            log_line = (f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | "
                        f"Recall: {val_recall:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.6f}\n")
            print(log_line, end="")
            log_file.write(log_line)

            # Early stopping
            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold}_epoch{epoch}.pt"))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered")
                    log_file.write("Early stopping triggered\n")
                    break

        logs_df = pd.DataFrame(logs)
        all_logs.append(logs_df)
        fold_scores.append(logs_df['val_f1'].max())


    # --------------------------
    # PLOT TRAINING METRICS PER FOLD
    # --------------------------
    plt.figure(figsize=(15,5))

    # Loss
    plt.subplot(1,3,1)
    plt.plot(logs_df['epoch'], logs_df['train_loss'], label='Train Loss', marker='o')
    plt.plot(logs_df['epoch'], logs_df['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1,3,2)
    plt.plot(logs_df['epoch'], logs_df['val_acc'], label='Val Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold} Accuracy')
    plt.grid(True)

    # F1 / Precision / Recall
    plt.subplot(1,3,3)
    plt.plot(logs_df['epoch'], logs_df['val_f1'], label='Val F1', marker='x', color='red')
    plt.plot(logs_df['epoch'], logs_df['val_precision'], label='Val Precision', marker='s', color='orange')
    plt.plot(logs_df['epoch'], logs_df['val_recall'], label='Val Recall', marker='^', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Fold {fold} Scores')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'training_metrics_fold{fold}_{MODEL_TYPE}.png'))
    # plt.show()

# --------------------------
# SUMMARY ACROSS FOLDS
# --------------------------
all_logs_df = pd.concat(all_logs, ignore_index=True)
fold_scores = np.array(fold_scores)

median_f1 = np.median(fold_scores)
best_f1 = fold_scores.max()
worst_f1 = fold_scores.min()
range_f1 = best_f1 - worst_f1

# --------------------------
# SUMMARY ACROSS FOLDS
# --------------------------
all_logs_df = pd.concat(all_logs, ignore_index=True)
fold_scores = np.array(fold_scores)

median_f1 = np.median(fold_scores)
best_f1 = fold_scores.max()
worst_f1 = fold_scores.min()
range_f1 = best_f1 - worst_f1

# Compute median/min/max/range for accuracy, precision, recall
val_accs = all_logs_df.groupby('fold')['val_acc'].max().values
val_precisions = all_logs_df.groupby('fold')['val_precision'].max().values
val_recalls = all_logs_df.groupby('fold')['val_recall'].max().values

median_acc = np.median(val_accs)
best_acc = val_accs.max()
worst_acc = val_accs.min()
range_acc = best_acc - worst_acc

median_precision = np.median(val_precisions)
best_precision = val_precisions.max()
worst_precision = val_precisions.min()
range_precision = best_precision - worst_precision

median_recall = np.median(val_recalls)
best_recall = val_recalls.max()
worst_recall = val_recalls.min()
range_recall = best_recall - worst_recall

print(f"\n--- Cross-validation Metrics ---")
print(f"F1: Median ± Range = {median_f1:.4f} ± {range_f1:.4f} (Best: {best_f1:.4f}, Worst: {worst_f1:.4f})")
print(f"Accuracy: Median ± Range = {median_acc:.4f} ± {range_acc:.4f} (Best: {best_acc:.4f}, Worst: {worst_acc:.4f})")
print(f"Precision: Median ± Range = {median_precision:.4f} ± {range_precision:.4f} (Best: {best_precision:.4f}, Worst: {worst_precision:.4f})")
print(f"Recall: Median ± Range = {median_recall:.4f} ± {range_recall:.4f} (Best: {best_recall:.4f}, Worst: {worst_recall:.4f})")

with open(log_txt_path, "a") as log_file:
    log_file.write(f"\n\n--- Cross-validation Metrics ---\n")
    log_file.write(f"F1: Median ± Range = {median_f1:.4f} ± {range_f1:.4f} (Best: {best_f1:.4f}, Worst: {worst_f1:.4f})\n")
    log_file.write(f"Accuracy: Median ± Range = {median_acc:.4f} ± {range_acc:.4f} (Best: {best_acc:.4f}, Worst: {worst_acc:.4f})\n")
    log_file.write(f"Precision: Median ± Range = {median_precision:.4f} ± {range_precision:.4f} (Best: {best_precision:.4f}, Worst: {worst_precision:.4f})\n")
    log_file.write(f"Recall: Median ± Range = {median_recall:.4f} ± {range_recall:.4f} (Best: {best_recall:.4f}, Worst: {worst_recall:.4f})\n")


# --------------------------
# PLOT F1 PER FOLD
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, K_FOLDS+1), fold_scores, marker='o', label='Fold Best F1')
plt.axhline(np.median(fold_scores), color='red', linestyle='--', label='Median F1')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('Best F1 per Fold')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, f'folds_f1_{MODEL_TYPE}.png'))
# plt.show()


# --------------------------
# PREDICTION (Test) CON MAJORITY VOTE
# --------------------------

label_map = {v: k for k, v in label_mapping.items()}

for m in models:
    m.eval()

all_preds = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)

        batch_preds = []
        for m in models:
            y_pred = m(X_batch)
            batch_preds.append(y_pred.argmax(1).cpu().numpy())

        batch_preds = np.stack(batch_preds, axis=1)  

        batch_majority = []
        for i in range(batch_preds.shape[0]):
            counts = np.bincount(batch_preds[i])
            batch_majority.append(np.argmax(counts))
        all_preds.extend(batch_majority)

pred_labels = [label_map[p] for p in all_preds]

sample_indices = X_test['sample_index'].drop_duplicates().reset_index(drop=True)
sample_indices_str = sample_indices.apply(lambda x: f"{x:03d}")
submission = pd.DataFrame({
    "sample_index": sample_indices_str,
    "label": pred_labels
})

submission_path = os.path.join(PRED_DIR, f"submission_{MODEL_TYPE}.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved at: {submission_path}")

# kaggle competitions submit -c an2dl2526c1 -f prediction/submission_{MODEL_TYPE}.csv -m "Majority vote across folds"
