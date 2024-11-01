import argparse
import os
import pickle
import sys,gc
import joblib,json
import numpy as np
import torch
import glob,random
from torch.nn import BCELoss,CrossEntropyLoss,LogSoftmax,NLLLoss
from torch.optim import Adam
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader
from model import GGNN_simplify, GCN_simplify2, DevignModel, IVDetect, DeepWukong, RevealModel,DeepWukong2
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def read_json(filename):
    #读取文件
    try:
        with open(filename.strip(),'r') as f:
            file = json.load(f)
    except:
        filename2=filename.strip().split('/')[-1]
        if int(filename2[0]) == 0:
            filename2='1' + filename2[2:]
            filename=filename.strip().split('0\'_')[0]+filename2
            with open(filename,'r') as f:
                file = json.load(f)
            file['target']=0
        else:
            filename2='0'+filename2[2:]
            filename=filename.strip().split('1\'_')[0]+filename2
            with open(filename,'r') as f:
                file = json.load(f)
            file['target']=1
    #文件内容读取到torch.tensor()中
    x = torch.tensor(file['node_features'],dtype=torch.float64)
    num_nodes = x.shape[0]

    edge_index_list = []
    for edge in file['graph']:
        # if edge[0] <= num_nodes and edge[2] <= num_nodes:
        if edge[0] <= num_nodes and edge[2] <= num_nodes:
            edge_index_list.append([edge[0],edge[2]])
            #edge_index_list.append([edge[0],edge[1]])
    edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()
    
    edge_attr_list = []
    for edge in file['graph']:
        edge_attr_list.append([edge[1]])
    edge_attr = torch.tensor(edge_attr_list)

    #y=[]
    #y.append([file['target']])
    #y=torch.tensor(y)
    y = torch.tensor([file['target']], dtype=int)
    data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name = filename.strip().split('/')[-1])
    #torch.save(data,filename+'.pt')
    return data

def save_gru(_trainLoader, model, device):
    ggnn_output = {}
    model.eval()
    with torch.no_grad():
        for data in tqdm(_trainLoader):
            if device != 'cpu':
                data = data.cuda()
            
            try:
                outputs = model(data.x.to(torch.float32), data.edge_index)
                outputs = outputs.detach().cpu().numpy()
                ggnn_output[data.name[0]] = outputs
            except:
                continue
    return ggnn_output

class DataSet_temp:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.read_dataset(test_src, train_src, valid_src)
        dataset = {'train': self.train_examples, 'val': self.valid_examples, 'test':  self.test_examples}
        self.dataset_loader=self.create_dataloader(dataset)
    

    def create_dataloader(self,dataset):
        fobatch=[]
        # for i in range(8):
        #     fobatch.append('n_'+str(i))
        loader = {'train': DataLoader(dataset['train'], batch_size=1, shuffle=True),
                  #'val': DataLoader(dataset['val'], batch_size=data_args.val_bs, shuffle=True),
                  #'test': DataLoader(dataset['test'], batch_size=1, shuffle=False),
                  #'explain': DataLoader(dataset['test'], batch_size=data_args.x_bs, shuffle=False)
        }
        return loader


    def read_dataset(self, test_src, train_src, valid_src):
        if train_src is not None:
            # print('Reading Train File!',train_src)
            # with open(train_src) as fp:
            #     path_list = fp.readlines()
                i = 0

                for path in tqdm(train_src):
                    #if i>50:
                    #    break
                    path=path.strip()
                    #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                    data_name = path
                    data = read_json(data_name)
                    # if(data.num_nodes >= 10):
                    #     i+=1
                    self.train_examples.append(data)
        else:
            path_list = glob.glob('/home/DT_devign/dataset/CPG_ast'+'/*.json')
            # with open('/home/DT_devign/data/devign/no_noise/no_noise_filter.txt','r') as rf:
            #     path_list=rf.readlines()
            # for index,item in enumerate(path_list):
            #     name = item.split('/')[-1]
            #     path_list[index]='/home/DT_devign/dataset/PDG_input/' + name
            i=0
            for path in tqdm(path_list):
                    # if i>50:
                    #    break
                    path=path.strip()
                    #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                    data_name = path
                    data = read_json(data_name)
                    if(data.num_nodes >= 7):
                        i+=1
                        self.train_examples.append(data)
        random.shuffle(self.train_examples)
        if valid_src is not None:
            print('Reading Valid File!',valid_src)
            with open(valid_src) as fp:
                path_list = fp.readlines()
                i = 0
                for path in tqdm(path_list):
                    #if i>7:
                    #    break
                    path=path.strip()
                    #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                    data_name = path
                    data = read_json(data_name)
                    if(data.num_nodes >= 10):
                        i+=1
                        self.valid_examples.append(data)
        random.shuffle(self.valid_examples)
        record_txt=[]
        if test_src is not None:
            print('Reading Test File!',test_src)
            with open(test_src) as fp:
                path_list = fp.readlines()
            #path_list = glob.glob(test_src+'/*.json')
            i = 0
            for path in tqdm(path_list):
                #if i>10:
                #    break
                path=path.strip()
                #data_name = '/home/nvd_dataset/'+path.split('/home/nvd/')[-1]
                data_name = path
                data = read_json(data_name)
                
                self.test_examples.append(data)
                record_txt.append(path)
            # with open("/home/GNNLRP_model/data/compltet_test.txt", 'w') as p_r:
            #     p_r.writelines(record_txt)
        random.shuffle(self.test_examples)

def calculate_distance(x1, y1, x2, y2):
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance


if __name__ == '__main__':
    noise_PDGs = glob.glob('/home/DT_devign/dataset/CPG_pdg'+'/*')
    pure_PDGs = glob.glob('/home/Real_Vul/CPG/CPG_PDG'+'/*')
    noise_PDG = []
    noise_AST = []
    noise_CFG = []
    num0=0
    num1=0
    # random.shuffle(noise_PDGs)
    for file in noise_PDGs:
        name = file.split('/')[-1]
        if name.startswith('0'):
            num0+=1
            if num0 <= 270:
                noise_PDG.append(file)
                AST_path = '/home/DT_devign/dataset/CPG_ast/'+name
                CFG_path = '/home/DT_devign/dataset/CPG_cfg/'+name
                noise_AST.append(AST_path)
                noise_CFG.append(CFG_path)
        if name.startswith('1'):
            num1+=1
            if num1 <= 270:
                noise_PDG.append(file)
                AST_path = '/home/DT_devign/dataset/CPG_ast/'+name
                CFG_path = '/home/DT_devign/dataset/CPG_cfg/'+name
                noise_AST.append(AST_path)
                noise_CFG.append(CFG_path)
        if num0>270 and num1>270:
            break
    num0=0
    num1=0
    for file in pure_PDGs:
        name = file.split('/')[-1]
        if name.startswith('0'):
            num0+=1
            if num0 <= 150:
                noise_PDG.append(file)
                AST_path = '/home/Real_Vul/CPG/CPG_AST/'+name
                CFG_path = '/home/Real_Vul/CPG/CPG_CFG/'+name
                noise_AST.append(AST_path)
                noise_CFG.append(CFG_path)
        if name.startswith('1'):
            num1+=1
            if num1 <= 150:
                noise_PDG.append(file)
                AST_path = '/home/Real_Vul/CPG/CPG_AST/'+name
                CFG_path = '/home/Real_Vul/CPG/CPG_CFG/'+name
                noise_AST.append(AST_path)
                noise_CFG.append(CFG_path)
        if num0>150 and num1>150:
            break
    processed_AST_noise_path = os.path.join('/home/DT_devign/motivation', 'AST_noise+200.bin')
    if True and os.path.exists(processed_AST_noise_path):
        print('*'*20)
        AST_dataset = joblib.load(open(processed_AST_noise_path, 'rb'))
        AST_dataloader = AST_dataset.dataset_loader
    else:
        AST_dataset = DataSet_temp(train_src = noise_AST,
                                    valid_src=None,
                                    test_src=None,
                                    )
        AST_dataloader = AST_dataset.dataset_loader
        file1 = open(processed_AST_noise_path, 'wb')
        joblib.dump(AST_dataset, file1)
        file1.close()

    processed_CFG_noise_path = os.path.join('/home/DT_devign/motivation', 'CFG_noise+200.bin')
    if True and os.path.exists(processed_CFG_noise_path):
        print('*'*20)
        CFG_dataset = joblib.load(open(processed_CFG_noise_path, 'rb'))
        CFG_dataloader = CFG_dataset.dataset_loader
    else:
        CFG_dataset = DataSet_temp(train_src = noise_CFG,
                                valid_src=None,
                                test_src=None,
                                )
        CFG_dataloader = CFG_dataset.dataset_loader
        file2 = open(processed_CFG_noise_path, 'wb')
        joblib.dump(CFG_dataset, file2)
        file2.close()

    processed_PDG_noise_path = os.path.join('/home/DT_devign/motivation', 'PDG_noise+200.bin')
    if True and os.path.exists(processed_PDG_noise_path):
        print('*'*20)
        PDG_dataset = joblib.load(open(processed_PDG_noise_path, 'rb'))
        PDG_dataloader = PDG_dataset.dataset_loader
    else:      
        PDG_dataset = DataSet_temp(train_src = noise_PDG,
                                valid_src=None,
                                test_src=None,
                                )
        PDG_dataloader = PDG_dataset.dataset_loader
        file3 = open(processed_PDG_noise_path, 'wb')
        joblib.dump(PDG_dataset, file3)
        file3.close()
    model = GGNN_simplify()
    model.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ggnn_AST_input = save_gru(AST_dataloader['train'], model , device)
    ggnn_CFG_input = save_gru(CFG_dataloader['train'], model , device)
    ggnn_PDG_input = save_gru(PDG_dataloader['train'], model , device)

    res_dis = 0
    dis_num = 0
    for key in tqdm(ggnn_AST_input):
        #pca = PCA(n_components=2)
        AST_features = ggnn_AST_input[key]
        if key in ggnn_CFG_input:
            CFG_features = ggnn_CFG_input[key]
        else:
            continue
        if key in ggnn_PDG_input:
            PDG_features = ggnn_PDG_input[key]
        else:
            continue
        try:
            # reduced_AST = TSNE(n_components=2,random_state=0,init='pca',perplexity=5).fit_transform(AST_features)
            reduced_CFG = TSNE(n_components=2,random_state=0,init='pca',perplexity=5).fit_transform(CFG_features)
            reduced_PDG = TSNE(n_components=2,random_state=0,init='pca',perplexity=5).fit_transform(PDG_features)
        except:
            continue
        #reduced = pca.fit_transform(slice_emb)
        dis_num += 1
        # t_AST = reduced_AST.transpose()
        # new_emb=np.array(np.mean(t_AST, axis=1))
        # AST_x = new_emb[0]
        # AST_y = new_emb[1]
        t_CFG = reduced_CFG.transpose()
        new_emb=np.array(np.mean(t_CFG, axis=1))
        CFG_x = new_emb[0]
        CFG_y = new_emb[1]
        t_PDG = reduced_PDG.transpose()
        new_emb=np.array(np.mean(t_PDG, axis=1))
        PDG_x = new_emb[0]
        PDG_y = new_emb[1]
        # dis1 = calculate_distance(AST_x,AST_y,CFG_x,CFG_y)
        # dis2 = calculate_distance(AST_x,AST_y,PDG_x,PDG_y)
        dis3 = calculate_distance(CFG_x,CFG_y,PDG_x,PDG_y)
        # dis_mean = (dis1+dis2+dis3)/3
        dis_mean = dis3
        res_dis+=dis_mean
    res_dis_mean = res_dis/dis_num
    print(res_dis_mean)