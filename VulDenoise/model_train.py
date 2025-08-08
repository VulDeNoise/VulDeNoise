import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv,GCNConv,global_mean_pool,global_max_pool, global_add_pool,TopKPooling,GlobalAttention

class Config:
    DATA_DIR = "/embedding/pdg_vec"
    BEST_MODEL_PATH = "/denoise/10/deepwukong"
    SPLIT_FILE_PATH = "ori_split_files.json"
    INPUT_DIM = 100
    HIDDEN_DIM = 200
    NUM_GGNN_STEPS = 4
    NUM_CLASSES = 2
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_SPLIT_RATIO = 0.2
    VALIDATION_SPLIT_RATIO = 0.1


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


def load_graph_from_json(filepath,flipped_labels_state):

    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    basename = os.path.basename(filepath)
    if flipped_labels_state:
        if basename in flipped_labels_state:
            original_label = data_dict['target']
            data_dict['target'] = 1 - original_label
    if data_dict['node_features'] == []:
        return None
    x = torch.tensor(data_dict['node_features'], dtype=torch.float32)
    edge_index_list = [[src, dst] for src, _, dst in data_dict['graph'] if src < len(x) and dst < len(x)]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([data_dict['target']], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, name=basename)

def load_dataset_from_filenames(data_dir, filenames,flip_list=None):
    
    dataset = []
    for fname in tqdm(filenames, desc="Loading data from files", leave=False):
        filepath = os.path.join(data_dir, fname)
        graph_data = load_graph_from_json(filepath,flip_list)
        if graph_data:
            dataset.append(graph_data)
    return dataset

def load_and_manage_splits_by_filename(data_dir, split_file_path,flip_list):
   
    if os.path.exists(split_file_path):
        print(f"find split_file '{split_file_path}', loading..")
        with open(split_file_path, 'r') as f:
            split_files = json.load(f)
        
        train_dataset = load_dataset_from_filenames(data_dir, split_files['train'],flip_list)
        val_dataset = load_dataset_from_filenames(data_dir, split_files['val'],flip_list)
        test_dataset = load_dataset_from_filenames(data_dir, split_files['test'],flip_list)

    else:
        print(f"no split_file，creat and save in '{split_file_path}'...")
        all_filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        all_filenames.sort()

        try:
            labels = [int(f.split('_')[0]) for f in all_filenames]
        except (ValueError, IndexError):
            print("waring:can 't get label from file name, use None")
            labels = None

        
        train_val_files, test_files = train_test_split(
            all_filenames, test_size=Config.TEST_SPLIT_RATIO, stratify=labels, random_state=2024
        )
        
        
        train_val_labels = None
        if labels:
            
            fname_to_label = dict(zip(all_filenames, labels))
            train_val_labels = [fname_to_label[f] for f in train_val_files]

        
        val_split_ratio_adjusted = Config.VALIDATION_SPLIT_RATIO / (1 - Config.TEST_SPLIT_RATIO)
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_split_ratio_adjusted, stratify=train_val_labels, random_state=2024
        )

        
        split_files_to_save = {'train': train_files, 'val': val_files, 'test': test_files}
        with open(split_file_path, 'w') as f:
            json.dump(split_files_to_save, f, indent=4)
        print("new split file has been saved")
        
        
        train_dataset = load_dataset_from_filenames(data_dir, train_files)
        val_dataset = load_dataset_from_filenames(data_dir, val_files)
        test_dataset = load_dataset_from_filenames(data_dir, test_files)
    
    print(f"dataset loaded：\n  - train: {len(train_dataset)}\n  - val: {len(val_dataset)}\n  - test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
   
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item()
            
            
            preds = logits.argmax(dim=1)
            
            
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    return avg_loss, all_labels, all_preds


if __name__ == '__main__':
    flip_list=None
    ### add Noise
    with open('it1/new_noise.json', 'r') as f:
        flip_list = json.load(f)
    
    train_data, val_data, test_data = load_and_manage_splits_by_filename(Config.DATA_DIR,Config.SPLIT_FILE_PATH,flip_list)

    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"\nuse devices: {Config.DEVICE}")
    # model = DevignModel(
    #     input_dim=Config.INPUT_DIM,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     num_steps=Config.NUM_GGNN_STEPS,
    #     num_classes=Config.NUM_CLASSES
    # ).to(Config.DEVICE)
    
    # model = RevealModel(
    #     input_dim=Config.INPUT_DIM,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     num_steps=Config.NUM_GGNN_STEPS,
    #     num_classes=Config.NUM_CLASSES
    # ).to(Config.DEVICE)
    
    # model = IVDetect(
    #     input_dim=Config.INPUT_DIM,
    #     output_dim=Config.HIDDEN_DIM,
    #     num_classes=Config.NUM_CLASSES
    # ).to(Config.DEVICE)
    
    model = DeepWukongModel(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- begin training ---")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_model_filename = None 
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        
       
        val_loss, val_labels, val_preds = evaluate(model, val_loader, criterion, Config.DEVICE)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, pos_label=1, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, pos_label=1, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"\nEpoch {epoch}/{Config.EPOCHS}:")
        print(f"  - training loss: {train_loss:.4f}")
        print(f"  - val loss: {val_loss:.4f}")
        print(f"  - val_Accuracy: {val_accuracy:.4f}")
        print(f"  - val_Precision: {val_precision:.4f}")
        print(f"  - val_Recall: {val_recall:.4f}")
        print(f"  - val_F1: {val_f1:.4f}")
        # print(f"  - 驗證macro_F1: {val_f1_macro:.4f}")
        
        if val_recall<0.8:
            new_filename = (f"f1_{val_f1:.4f}_acc_{val_accuracy:.4f}_"
                                f"pre_{val_precision:.4f}_rec_{val_recall:.4f}.pt")
            save_path = os.path.join(Config.BEST_MODEL_PATH, new_filename)
            
            torch.save(model.state_dict(), save_path)
    
        if val_accuracy > best_val_acc and val_recall<0.75 :
            best_val_acc = val_accuracy
            # if best_model_filename and os.path.exists(best_model_filename):
            #     os.remove(best_model_filename)

            # new_filename = (f"f1_{val_f1:.4f}_acc_{val_accuracy:.4f}_"
            #                 f"pre_{val_precision:.4f}_rec_{val_recall:.4f}.pt")
            # save_path = os.path.join(Config.BEST_MODEL_PATH, new_filename)
            
            # torch.save(model.state_dict(), save_path)
            best_model_filename = save_path
            print(f"  - new best model has been saved in: {os.path.basename(save_path)}")
            
    print("\n--- training finished，loading best model and evaluate in test set ---")
    
    if best_model_filename:
        print(f"loading best model: {os.path.basename(best_model_filename)}")
        model.load_state_dict(torch.load(best_model_filename))
        
        test_loss, test_labels, test_preds = evaluate(model, test_loader, criterion, Config.DEVICE)
        
        report = classification_report(
            test_labels, test_preds, 
            target_names=[f"Class {i}" for i in range(Config.NUM_CLASSES)], 
            digits=4
        )
        
        print(f"\ntest res:\n  - test loss: {test_loss:.4f}\n\n classification report:\n{report}")
    else:
        print("no best model, skip test set evaluation")