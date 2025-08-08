# track_loss_only.py

import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv,GCNConv,TopKPooling,GlobalAttention, global_add_pool, global_mean_pool,global_max_pool # 使用一個模型作為示例

class Config:
    DATA_DIR = "ast_cfg_pdg_vec(save path)"
    SPLIT_FILE_PATH = "ori_split_files.json"
    LOSS_HISTORY_PATH = "ivdetect/50/AST_loss.json"

    INPUT_DIM, HIDDEN_DIM, NUM_CLASSES = 100, 200, 2
    NUM_GGNN_STEPS = 4
    LEARNING_RATE, EPOCHS, BATCH_SIZE = 0.0001, 50, 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO = 0.2, 0.1

class IVDetect(nn.Module):

    def __init__(self, output_dim, input_dim, num_classes):
        super().__init__()
        self.out_dim = output_dim #200
        self.in_dim = input_dim
        self.conv1 = GCNConv(input_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        self.conv3 = GCNConv(output_dim, num_classes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)##0.3
        self.connect = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch):
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv,edge_index))
        post_conv = self.conv3(post_conv,edge_index)
        pooled = global_max_pool(post_conv, batch)
        return pooled

class RevealModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, dropout_rate=0.2, **kwargs):
        super().__init__()

        
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_steps)
        mlp_hidden_dim = 2 * hidden_dim

        self.feature_extractor_mlp = nn.Sequential(
            
            nn.Linear(in_features=hidden_dim, out_features=mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(in_features=mlp_hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(in_features=hidden_dim, out_features=mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.classifier = nn.Linear(in_features=mlp_hidden_dim, out_features=num_classes)


    def forward(self, x, edge_index, batch):
        
        node_embeddings = self.ggnn(x, edge_index)
        
        
        graph_embedding = global_add_pool(node_embeddings, batch)
        
        
        graph_features = self.feature_extractor_mlp(graph_embedding)
        
        
        logits = self.classifier(graph_features)
        
        return logits

class DeepWukongModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5, top_k_ratio=0.8, **kwargs):
       
        super().__init__()

        
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.pooling = TopKPooling(hidden_dim, ratio=top_k_ratio)
        
        self.attention_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        
        mlp_hidden_dim = 2 * hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.classifier = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        
        
        node_embedding = self.gcn_conv(x, edge_index)
        node_embedding = F.relu(node_embedding)

        
        node_embedding, edge_index, _, batch, _, _ = self.pooling(
            x=node_embedding,
            edge_index=edge_index,
            batch=batch
        )

       
        graph_embedding = self.attention_pool(
            x=node_embedding,
            batch=batch
        )
        
        
        hidden_features = self.mlp(graph_embedding)
        
       
        logits = self.classifier(hidden_features)
        
        return logits
    
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



def load_graph_from_json(filepath,flipped_labels_state):
    basename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f: data_dict = json.load(f)
    except FileNotFoundError: return None
    if flipped_labels_state:
        if basename in flipped_labels_state:
            original_label = data_dict['target']
            data_dict['target'] = 1 - original_label
    x = torch.tensor(data_dict['node_features'], dtype=torch.float32)
    edge_index_list = [[src, dst] for src, _, dst in data_dict['graph'] if src < len(x) and dst < len(x)]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([data_dict['target']], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, name=basename)

def load_dataset_from_filenames(data_dir, filenames,flip_list):
    return [data for fname in tqdm(filenames, desc=f"Loading {len(filenames)} files", leave=False) if (data := load_graph_from_json(os.path.join(data_dir, fname),flip_list))]

def load_and_manage_splits_by_filename(data_dir, split_file_path,flip_list=None):
    if os.path.exists(split_file_path):
        print(f"find split_file '{split_file_path}',loading...")
        with open(split_file_path, 'r') as f: split_files = json.load(f)
        train_dataset = load_dataset_from_filenames(data_dir, split_files['train'], flip_list)
        # val_dataset = load_dataset_from_filenames(data_dir, split_files['val'])
        # test_dataset = load_dataset_from_filenames(data_dir, split_files['test'])
    else:
        print(f"no split_file，create new split_file '{split_file_path}'...")
        all_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        try:
            labels = [int(f.split('_')[0]) for f in all_filenames]
        except (ValueError, IndexError): labels = None
        train_val_files, test_files = train_test_split(all_filenames, test_size=Config.TEST_SPLIT_RATIO, stratify=labels, random_state=42)
        train_val_labels = None
        if labels:
            fname_to_label = dict(zip(all_filenames, labels))
            train_val_labels = [fname_to_label[f] for f in train_val_files]
        val_split_ratio_adjusted = Config.VALIDATION_SPLIT_RATIO / (1 - Config.TEST_SPLIT_RATIO)
        train_files, val_files = train_test_split(train_val_files, test_size=val_split_ratio_adjusted, stratify=train_val_labels, random_state=42)
        split_files_to_save = {'train': train_files, 'val': val_files, 'test': test_files}
        with open(split_file_path, 'w') as f: json.dump(split_files_to_save, f, indent=4)
        print("new split_file has been saved.")
        train_dataset = load_dataset_from_filenames(data_dir, train_files)

    print(f"training set size: {len(train_dataset)}")
    return train_dataset


def train_one_epoch(model, loader, optimizer, criterion, loss_history, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        
        per_sample_losses = criterion(logits, batch.y)
        
        for i, name in enumerate(batch.name):
            loss_history[name].append(per_sample_losses[i].item())
            
        loss = per_sample_losses.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':
    flip_list=None
    #iteration reference
    # with open('first_epoch_noise_pred.json', 'r') as f:
    #     flip_list = json.load(f)
    
    train_data = load_and_manage_splits_by_filename(
        data_dir=Config.DATA_DIR, 
        split_file_path=Config.SPLIT_FILE_PATH,
        flip_list=flip_list
    )

    loss_history = {data.name: [] for data in train_data}

    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    print(f"\nusing devices: {Config.DEVICE}")
    
    # model = DevignModel(
    #     input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM,
    #     num_classes=Config.NUM_CLASSES, num_steps=Config.NUM_GGNN_STEPS
    # ).to(Config.DEVICE)
    
    # model = RevealModel(
    #     input_dim=Config.INPUT_DIM,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     num_steps=Config.NUM_GGNN_STEPS,
    #     num_classes=Config.NUM_CLASSES
    # ).to(Config.DEVICE)
    
    model = IVDetect(
        input_dim=Config.INPUT_DIM,
        output_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # model = DeepWukongModel(
    #     input_dim=Config.INPUT_DIM,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     num_classes=Config.NUM_CLASSES
    # ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    criterion_train = nn.CrossEntropyLoss(reduction='none')
    
    print("\n--- begin training and record training loss ---")
    
    for epoch in range(1, Config.EPOCHS + 1):
        # training and update loss_history
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion_train, loss_history, Config.DEVICE)
        print(f"Epoch {epoch}/{Config.EPOCHS} finished, mean training loss: {avg_train_loss:.4f}")
            
    print("\n--- training finished---")

    # 保存最終的損失歷史記錄
    print(f"\nrecord every sample 's loss and save in '{Config.LOSS_HISTORY_PATH}'...")
    with open(Config.LOSS_HISTORY_PATH, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print("loss history has been saved.")