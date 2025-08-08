# cross_validation_pipeline.py

import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_add_pool, global_mean_pool

class Config:
    DATA_DIR = "pdg_path"
    SPLIT_FILE_PATH = "ori_split_files.json"
    NUM_FOLDS = 5
    INPUT_DIM, HIDDEN_DIM, NUM_CLASSES = 100, 200, 2
    NUM_GGNN_STEPS = 6
    LEARNING_RATE, EPOCHS, BATCH_SIZE = 0.001, 100, 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RevealModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, gnn_steps, dropout_rate=0.2, **kwargs):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=gnn_steps)
        mlp_hidden_dim = 2 * hidden_dim
        self.feature_extractor_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.classifier = nn.Linear(mlp_hidden_dim, num_classes)
    def forward(self, x, edge_index, batch):
        node_embeddings = self.ggnn(x, edge_index)
        graph_embedding = global_add_pool(node_embeddings, batch)
        graph_features = self.feature_extractor_mlp(graph_embedding)
        return self.classifier(graph_features)

class DevignModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(out_channels=self.hidden_dim, num_layers=self.num_timesteps)
        self.concat_dim = self.input_dim + self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.concat_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        node_embeddings = self.ggnn(x, edge_index)
        concatenated_features = torch.cat((x, node_embeddings), dim=1)
        graph_embedding = global_mean_pool(concatenated_features, batch)
        logits = self.mlp(graph_embedding)
        return logits


def load_graph_from_json(filepath):
    basename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f: data_dict = json.load(f)
    except FileNotFoundError: return None
    x = torch.tensor(data_dict['node_features'], dtype=torch.float32)
    edge_index_list = [[src, dst] for src, _, dst in data_dict['graph'] if src < len(x) and dst < len(x)]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([data_dict['target']], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, name=basename)

def load_dataset_from_filenames(data_dir, filenames):
    return [data for fname in tqdm(filenames, desc=f"Loading {len(filenames)} files", leave=False) if (data := load_graph_from_json(os.path.join(data_dir, fname)))]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
    return accuracy_score(all_labels, all_preds)

if __name__ == '__main__':
    
    if not os.path.exists(Config.SPLIT_FILE_PATH):
        print(f"error: can't find split_file '{Config.SPLIT_FILE_PATH}'。")
        exit()

    print("loading...")
    with open(Config.SPLIT_FILE_PATH, 'r') as f:
        split_files = json.load(f)
    
    cv_pool_filenames = split_files['train'] + split_files['val']
    cv_dataset = load_dataset_from_filenames(Config.DATA_DIR, cv_pool_filenames)
    
   
    X_indices = np.arange(len(cv_dataset))
    y = np.array([data.y.item() for data in cv_dataset])
    
    kfold = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f"\n--- begin {Config.NUM_FOLDS} -fold cross val---")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_indices, y)):
        print(f"\n===== FOLD {fold + 1}/{Config.NUM_FOLDS} =====")
        

        train_fold_dataset = [cv_dataset[i] for i in train_idx]
        val_fold_dataset = [cv_dataset[i] for i in val_idx]
        

        train_loader = DataLoader(train_fold_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_fold_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print("loading...")
        model = DevignModel(
            input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES, num_steps=Config.NUM_GGNN_STEPS
        ).to(Config.DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        print(f" training{Config.EPOCHS} ...")
        for epoch in tqdm(range(1, Config.EPOCHS + 1), desc=f"Fold {fold+1} Training"):
            train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        
        print("evaluate current fold...")
        fold_accuracy = evaluate(model, val_loader, Config.DEVICE)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} accuracy: {fold_accuracy:.4f}")

    print("\n--- fininshed ---")
    
    accuracies_np = np.array(fold_accuracies)
    mean_accuracy = accuracies_np.mean()
    std_accuracy = accuracies_np.std()
    
    print("\n confirm results:")
    for i, acc in enumerate(fold_accuracies):
        print(f"  - Fold {i+1} Accuracy: {acc:.4f}")
    
    print(f"\nmean acc: {mean_accuracy:.4f} ± {std_accuracy:.4f}")