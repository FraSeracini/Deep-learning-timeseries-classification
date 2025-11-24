# ==========================
# Pirate Pain Prediction: LSTM / GRU / Transformer / CNN1D / MLP / GNN / Ensemble Models
# ==========================

import json
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from scipy.ndimage import gaussian_filter1d
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import GroupKFold  # CAMBIATO: GroupKFold invece di StratifiedGroupKFold


# --------------------------
# CONFIG & SEEDS
# --------------------------
SEED = 5555656
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
CHECKPOINT_DIR = os.path.join(MODELS_DIR, timestamp, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------------------
# HYPERPARAMETERS
# --------------------------
MODEL_TYPE = "GRU"  # options: LSTM / GRU / Transformer / CNN1D / MLP / Ensemble / GNN
BATCH_SIZE = 64
NUM_EPOCHS = 70
LR = 1e-3
VALID_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 20
PREPROCESSING_VISUAL_CHECK = False

PREPROCESSING_OPTS = {
    "gaussian": "none",   # "gaussian", "min", "min+log", "min+asinh", "boxcox", "none"
    "exp": "min+asinh",
    "pain": "none"
}

SCALER_OPTS = {
    "gaussian": "standard",   # "standard", "minmax", "robust", "none"
    "exp": "none",
    "pain": "none"
}

epsilon = 1e-6

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
        min_val = x_train.min()
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

# --------------------------
# PIPELINE
# --------------------------

# Gaussian
# X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, gaussian_joints, PREPROCESSING_OPTS["gaussian"])
# X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, gaussian_joints, SCALER_OPTS["gaussian"])

# Exponential
# X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, exp_joints, PREPROCESSING_OPTS["exp"])
# X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, exp_joints, SCALER_OPTS["exp"])

# Pain survey integers
# X_train_enc, X_test_enc = apply_preprocessing(X_train_enc, X_test_enc, pain_cols, PREPROCESSING_OPTS["pain"])
# X_train_enc, X_test_enc = apply_scaler(X_train_enc, X_test_enc, pain_cols, SCALER_OPTS["pain"])

# Final feature columns
feature_cols = pain_cols + gaussian_joints + exp_joints + onehot_cols

# ===== GLOBAL PREPROCESSING PER CV =====

X_train_full_for_cv = pd.get_dummies(X_train, columns=categorical_cols)
X_test_full_for_cv = pd.get_dummies(X_test, columns=categorical_cols)
X_test_full_for_cv = X_test_full_for_cv.reindex(columns=X_train_full_for_cv.columns, fill_value=0)

for c in ['joint_30']:
    if c in X_train_full_for_cv.columns:
        X_train_full_for_cv.drop(columns=[c], inplace=True)
        X_test_full_for_cv.drop(columns=[c], inplace=True)

# global_scaler
global_scaler = StandardScaler()
global_scaler.fit(X_train_full_for_cv[feature_cols])

label_mapping = {label: idx for idx, label in enumerate(sorted(y_train['label'].unique()))}
y_train['label_encoded'] = y_train['label'].map(label_mapping)

# --------------------------
# FUNCTION prepare_cv_data 
# --------------------------

def prepare_cv_data(train_ids, val_ids):
    """Prepara dati per CV senza leakage"""
    tr_df = X_train[X_train['sample_index'].isin(train_ids)].copy()
    va_df = X_train[X_train['sample_index'].isin(val_ids)].copy()
    
    # One-hot encoding 
    tr_enc = pd.get_dummies(tr_df, columns=categorical_cols)
    va_enc = pd.get_dummies(va_df, columns=categorical_cols)
    va_enc = va_enc.reindex(columns=tr_enc.columns, fill_value=0)

    for c in ['joint_30']:
        if c in tr_enc.columns:
            tr_enc.drop(columns=[c], inplace=True)
            va_enc.drop(columns=[c], inplace=True)

    # Global preprocessing
    tr_pp = tr_enc.copy()
    va_pp = va_enc.copy()
    
    tr_pp[feature_cols] = global_scaler.transform(tr_enc[feature_cols])
    va_pp[feature_cols] = global_scaler.transform(va_enc[feature_cols])
    
    return tr_pp, va_pp

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
# SEQUENCE DATASET
# --------------------------
class PiratePainSeqDataset(Dataset):
    def __init__(self, X, y=None, feature_cols=None):
        self.X, self.y = [], []
        self.sample_ids = []

        if feature_cols is None:
            onehot_cols_fold = [c for c in X.columns if any(k in c for k in ['n_legs_', 'n_hands_', 'n_eyes_'])]
            base_cols_fold   = [c for c in (pain_cols + gaussian_joints + exp_joints) if c in X.columns]
            feature_cols = base_cols_fold + onehot_cols_fold
        self.feature_cols = feature_cols

        y_map = None
        if y is not None:
            y_map = y.set_index('sample_index')['label'].map(label_mapping)

        for idx, df_sub in X.groupby('sample_index'):
            seq = df_sub[self.feature_cols].values.astype(np.float32)
            self.X.append(seq)
            self.sample_ids.append(idx)
            if y_map is not None:
                if idx not in y_map.index:
                    raise ValueError(f"sample_index {idx} non trovato in y_train")
                self.y.append(y_map.loc[idx])

        print(f"\nTotal sequences loaded: {len(self.X)}")
        if y is not None:
            self.y = np.array(self.y)
            print(f"Total labels loaded: {len(self.y)}")
        else:
            self.y = None

    def __len__(self): return len(self.X)

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

# ===== Make_fold_loaders =====
def make_fold_loaders(train_ids, val_ids):
    tr_pp, va_pp = prepare_cv_data(train_ids, val_ids)
    
    onehot_cols_fold = [c for c in tr_pp.columns if any(k in c for k in ['n_legs_', 'n_hands_', 'n_eyes_'])]
    base_cols_fold = [c for c in (pain_cols + gaussian_joints + exp_joints) if c in tr_pp.columns]
    feature_cols_fold = base_cols_fold + onehot_cols_fold
    input_dim_fold = len(feature_cols_fold)

    ds_tr = PiratePainSeqDataset(tr_pp, y_train, feature_cols=feature_cols_fold)
    ds_va = PiratePainSeqDataset(va_pp, y_train, feature_cols=feature_cols_fold)

    train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, input_dim_fold

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

# CNN1D + SE BLOCK
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

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock1D(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels, kernel_sizes, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = input_dim
        for out_channels, k in zip(conv_channels, kernel_sizes):
            layers.append(ResidualBlock1D(in_channels, out_channels, k, dropout))
            in_channels = out_channels
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)

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
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(dim=1) 
        return self.net(x)
    

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

# --------------------------
# ENSEMBLE MODEL
# --------------------------
class EnsembleModel(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.device = device
        
        # --- CNN1D ---
        self.cnn = CNN1DModel(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=[48, 96, 160],  
            kernel_sizes=[5, 3, 3],
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

        # trainable weights
        self.model_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        """
        Forward comune (media pesata dei logits)
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
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(F.softmax(logits, dim=-1), dim=-1)

    def predict_majority(self, x, confidence_threshold=0.5):
        with torch.no_grad():
            preds = [
                torch.argmax(F.softmax(self.cnn(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.gru(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.lstm(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.transformer(x), dim=-1), dim=-1),
                torch.argmax(F.softmax(self.mlp(x), dim=-1), dim=-1)
            ]
            preds = torch.stack(preds, dim=1)  # [batch, 5]
            
            majority, counts = torch.mode(preds, dim=1)
            confident = (counts.float() / preds.size(1)) >= confidence_threshold
            fallback_mask = ~confident

            if fallback_mask.any():
                weighted_preds = self.predict(x[fallback_mask])
                majority[fallback_mask] = weighted_preds

            return majority


def build_model_by_type(cfg=None, input_dim=None, num_classes_arg=None):
    if cfg is None: cfg = {}
    mtype = cfg.get("type", MODEL_TYPE)
    if num_classes_arg is None:
        num_classes_arg = len(label_mapping)

    if input_dim is None:
        raise ValueError("build_model_by_type richiede input_dim per costruire i layer corretti.")

    if mtype=="LSTM":
        return LSTMModel(input_dim, cfg.get("hidden_size",192), cfg.get("num_layers",3),
                         num_classes_arg, cfg.get("dropout",0.4)).to(device)
    if mtype=="GRU":
        return GRUModel(input_dim, cfg.get("hidden_size",192), cfg.get("num_layers",3),
                        num_classes_arg, cfg.get("dropout",0.4)).to(device)
    if mtype=="Transformer":
        return TransformerModel(input_dim, num_classes_arg, cfg.get("hidden_dim",96),
                                cfg.get("nhead",2), cfg.get("num_layers",3),
                                500, cfg.get("dropout",0.3)).to(device)
    if mtype=="CNN1D":
        return CNN1DModel(input_dim, num_classes_arg, cfg.get("conv_channels",[48,96,160]),
                          cfg.get("kernel_sizes",[5,3,3]), cfg.get("dropout",0.4)).to(device)
    if mtype=="MLP":
        return MLPModel(input_dim, cfg.get("hidden_sizes",[384,192,128]),
                        num_classes_arg, cfg.get("dropout",0.4)).to(device)
    raise ValueError("MODEL_TYPE non supportato")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train(); tot=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        tot += loss.item()
    return tot/len(loader)

def eval_f1(model, loader, criterion):
    model.eval(); vloss=0.0; ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            vloss += criterion(logits, yb).item()
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(yb.cpu().numpy())
    vloss /= len(loader)
    yhat = np.concatenate(ps); ytrue = np.concatenate(ys)
    f1 = f1_score(ytrue, yhat, average='macro', zero_division=0)
    return vloss, f1

# ===== GroupKFold =====
seq_ids = np.array(sorted(X_train['sample_index'].unique()))
y_seq   = y_train.set_index('sample_index').loc[seq_ids, 'label'].map(label_mapping).values
groups  = seq_ids.copy()

K = 10
gkf = GroupKFold(n_splits=K) 

# ===== Paramters Grids =====
all_param_grid = [
    # ---------- MLP ----------
    {"type": "MLP", "hidden_sizes": [256,128],       "dropout": 0.3, "lr": 1e-3},
    {"type": "MLP", "hidden_sizes": [384,192,128],   "dropout": 0.4, "lr": 1e-3},
    {"type": "MLP", "hidden_sizes": [512,256,128],   "dropout": 0.4, "lr": 1e-3},
    {"type": "MLP", "hidden_sizes": [512,256,128,64],"dropout": 0.5, "lr": 1e-3},
    {"type": "MLP", "hidden_sizes": [256,128],       "dropout": 0.3, "lr": 5e-4},

    # ---------- GRU ----------
    {"type": "GRU", "hidden_size": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
    {"type": "GRU", "hidden_size": 192, "num_layers": 3, "dropout": 0.4, "lr": 1e-3},
    {"type": "GRU", "hidden_size": 256, "num_layers": 2, "dropout": 0.4, "lr": 5e-4},
    {"type": "GRU", "hidden_size": 256, "num_layers": 3, "dropout": 0.5, "lr": 1e-3},
    {"type": "GRU", "hidden_size": 320, "num_layers": 3, "dropout": 0.4, "lr": 5e-4},

    # ---------- LSTM ----------
    {"type": "LSTM", "hidden_size": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
    {"type": "LSTM", "hidden_size": 192, "num_layers": 3, "dropout": 0.4, "lr": 1e-3},
    {"type": "LSTM", "hidden_size": 256, "num_layers": 2, "dropout": 0.4, "lr": 5e-4},
    {"type": "LSTM", "hidden_size": 320, "num_layers": 3, "dropout": 0.4, "lr": 5e-4},
    {"type": "LSTM", "hidden_size": 192, "num_layers": 4, "dropout": 0.5, "lr": 1e-3},

    # ---------- CNN1D ----------
    {"type": "CNN1D", "conv_channels": [32,64,128],     "kernel_sizes": [5,3,3], "dropout": 0.3, "lr": 1e-3},
    {"type": "CNN1D", "conv_channels": [48,96,160],     "kernel_sizes": [5,3,3], "dropout": 0.4, "lr": 1e-3},
    {"type": "CNN1D", "conv_channels": [64,128,256],    "kernel_sizes": [5,3,3], "dropout": 0.5, "lr": 5e-4},
    {"type": "CNN1D", "conv_channels": [64,128,256,512],"kernel_sizes": [7,5,3,3], "dropout": 0.4, "lr": 5e-4},

    # ---------- TRANSFORMER ----------
    {"type": "Transformer", "hidden_dim": 96,  "nhead": 2, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},
    {"type": "Transformer", "hidden_dim": 128, "nhead": 4, "num_layers": 3, "dropout": 0.4, "lr": 5e-4},
    {"type": "Transformer", "hidden_dim": 192, "nhead": 4, "num_layers": 4, "dropout": 0.4, "lr": 5e-4},
    {"type": "Transformer", "hidden_dim": 256, "nhead": 8, "num_layers": 4, "dropout": 0.4, "lr": 3e-4},
    {"type": "Transformer", "hidden_dim": 128, "nhead": 4, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
]

# ----- FILTRO FOR MODEL -----
param_grid = [cfg for cfg in all_param_grid if cfg["type"] == MODEL_TYPE]

print(f"\nTuning per MODEL_TYPE = {MODEL_TYPE}")
print(f"{len(param_grid)} configurazioni trovate.")

best_cfg, best_mean_f1 = None, -1.0
for cfg in param_grid:
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(seq_ids, y_seq, groups=groups)):
        train_ids = seq_ids[tr_idx]
        val_ids   = seq_ids[va_idx]
        train_loader, val_loader, input_dim_fold = make_fold_loaders(train_ids, val_ids)

        model = build_model_by_type(cfg, input_dim=input_dim_fold)
        lr = cfg.get("lr", LR if cfg["type"]!="Transformer" else LR*0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3,
                                      patience=(3 if cfg["type"]=="Transformer" else 5), min_lr=1e-7)

        best_vloss, wait, best_state = float('inf'), 0, None
        for epoch in range(1, NUM_EPOCHS+1):
            _ = train_one_epoch(model, train_loader, optimizer, criterion)
            vloss, vf1 = eval_f1(model, val_loader, criterion)
            scheduler.step(vloss)
            if vloss < best_vloss - 1e-4:
                best_vloss, wait = vloss, 0
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= EARLY_STOPPING_PATIENCE:
                    break

        model.load_state_dict(best_state)
        _, vf1 = eval_f1(model, val_loader, criterion)
        fold_scores.append(vf1)
        print(f"[{cfg}] Fold {fold+1}/{K} F1={vf1:.4f}")

    mean_f1 = float(np.mean(fold_scores))
    print(f"CFG {cfg} => mean macro-F1: {mean_f1:.4f}")
    if mean_f1 > best_mean_f1:
        best_mean_f1, best_cfg = mean_f1, cfg

print(f"\n>>> BEST CFG: {best_cfg} | mean macro-F1={best_mean_f1:.4f}")

# --- save best cfg ---
os.makedirs(os.path.join(MODELS_DIR, timestamp), exist_ok=True)
with open(os.path.join(MODELS_DIR, timestamp, "best_cfg.json"), "w") as f:
    json.dump(best_cfg, f, indent=2)

# ===== FINAL: fit preprocessing =====
# One-hot global (fit sul train intero)
X_train_full_enc = pd.get_dummies(X_train, columns=categorical_cols)
X_test_full_enc  = pd.get_dummies(X_test,  columns=categorical_cols)
X_test_full_enc  = X_test_full_enc.reindex(columns=X_train_full_enc.columns, fill_value=0)

for c in ['joint_30']:
    if c in X_train_full_enc.columns:
        X_train_full_enc.drop(columns=[c], inplace=True)
        X_test_full_enc.drop(columns=[c], inplace=True)

# final preprocessing 
X_train_full_pp, X_test_full_pp = apply_preprocessing(X_train_full_enc, X_test_full_enc, gaussian_joints, PREPROCESSING_OPTS["gaussian"])
X_train_full_pp, X_test_full_pp = apply_scaler(X_train_full_pp, X_test_full_pp, gaussian_joints, SCALER_OPTS["gaussian"])

X_train_full_pp, X_test_full_pp = apply_preprocessing(X_train_full_pp, X_test_full_pp, exp_joints, PREPROCESSING_OPTS["exp"])
X_train_full_pp, X_test_full_pp = apply_scaler(X_train_full_pp, X_test_full_pp, exp_joints, SCALER_OPTS["exp"])

X_train_full_pp, X_test_full_pp = apply_preprocessing(X_train_full_pp, X_test_full_pp, pain_cols, PREPROCESSING_OPTS["pain"])
X_train_full_pp, X_test_full_pp = apply_scaler(X_train_full_pp, X_test_full_pp, pain_cols, SCALER_OPTS["pain"])

onehot_cols_full = [c for c in X_train_full_pp.columns if any(k in c for k in ['n_legs_', 'n_hands_', 'n_eyes_'])]
base_cols_full   = [c for c in (pain_cols + gaussian_joints + exp_joints) if c in X_train_full_pp.columns]
feature_cols_full = base_cols_full + onehot_cols_full
input_dim = len(feature_cols_full)          
num_classes = len(label_mapping)

input_dim_full = len(feature_cols_full)
num_classes    = len(label_mapping)

final_train_ds = PiratePainSeqDataset(X_train_full_pp, y_train, feature_cols=feature_cols_full)
test_dataset   = PiratePainSeqDataset(X_test_full_pp,               feature_cols=feature_cols_full)


final_train_loader = DataLoader(final_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader        = DataLoader(test_dataset,   batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ===== Train model with best_cfg =====
model = build_model_by_type(best_cfg, input_dim=input_dim_full, num_classes_arg=num_classes)
lr = best_cfg.get("lr", LR if best_cfg["type"]!="Transformer" else LR*0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ---- FINAL FIT with logging + plot ----
final_logs = []
for epoch in range(1, NUM_EPOCHS+1):
    train_loss_epoch = train_one_epoch(model, final_train_loader, optimizer, criterion)
    final_logs.append({"epoch": epoch, "train_loss": train_loss_epoch})
    if epoch % 5 == 0:
        print(f"[FINAL FIT] Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss_epoch:.4f}")


if len(final_logs) > 0:
    logs_df = pd.DataFrame(final_logs)
    plt.figure(figsize=(7,4))
    plt.plot(logs_df['epoch'], logs_df['train_loss'], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Train Loss')
    plt.title('Final Fit – Train Loss')
    plt.grid(True); plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, timestamp, "final_fit_train_loss.png")
    plt.savefig(plot_path); plt.show()
    print(f" Final-fit plot salvato in: {plot_path}")

# --------------------------
# HYPERPARAMETERS
# --------------------------

hyperparameters = {
    "MODEL_TYPE": MODEL_TYPE,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "LR": LR,
    "VALID_SPLIT": VALID_SPLIT,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "PREPROCESSING_OPTS": PREPROCESSING_OPTS,
    "SCALER_OPTS": SCALER_OPTS,
    "epsilon": epsilon,
}

hyperparams_path = os.path.join(MODELS_DIR, timestamp, f"hyperparameters.json")
with open(hyperparams_path, 'w') as f:
    json.dump(hyperparameters, f, indent=4)
print(f" Hyperparameters saved at: {hyperparams_path}")

# --------------------------
# PREDICTION (Test)
# --------------------------

import os
import pandas as pd
import numpy as np
import torch

label_map = {v: k for k, v in label_mapping.items()}

model.eval()
preds = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        preds.append(y_pred.argmax(1).cpu().numpy())

preds = np.concatenate(preds)
pred_labels = [label_map[p] for p in preds]
sample_indices = X_test['sample_index'].drop_duplicates().reset_index(drop=True)
sample_indices_str = sample_indices.apply(lambda x: f"{x:03d}")
submission = pd.DataFrame({
    "sample_index": sample_indices_str,
    "label": pred_labels
})

submission_path = os.path.join(PRED_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved at: {submission_path}")
# kaggle competitions submit -c an2dl2526c1 -f prediction/submission.csv -m " "