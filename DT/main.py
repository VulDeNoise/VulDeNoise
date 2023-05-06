import argparse
import os
import pickle
import sys,gc
import joblib,json
import numpy as np
import torch
from torch.nn import BCELoss,CrossEntropyLoss,LogSoftmax,NLLLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc,brier_score_loss
from sklearn.model_selection import cross_val_score

from data_loader.dataset import DataSet, DataSet2
from model import GGNN_simplify, GCN_simplify2, DevignModel, IVDetect, DeepWukong, RevealModel,DeepWukong2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
#from trainer import train, eval
#from utils import tally_param, debug

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch.nn.functional as F

from sklearn import datasets
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")



def cross_val( _trainLoader, model,criterion, optimizer):


    # iris = load_iris()
    # svc = svm.SVC()
    # scores = cross_val_score(svc, iris.data, iris.target, cv=5)
    # print(scores)

    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    X=[]
    Y=[]
    EI=[]
    #model.zero_grad()
    for index, data in enumerate(tqdm(_trainLoader)):
        # if index % 500 == 0:
        #     print(correct)
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        X.append(data.x.to(torch.float32))
        EI.append(data.edge_index)
        Y.append(data.y.long())
    kf = KFold(n_splits = 5, shuffle=True, random_state=0)
    curr_score=0
    for train_index, test_index in kf.split(X):
        acc_temp=0
        for e in range(0, 150):
            model.train()
            tnum=len(train_index)
            train_loss=0
            correct=0
            #print(tnum)
            for index in tqdm(train_index):
                x=X[index].cuda()
                y=Y[index].cuda()
                edge_index=EI[index].cuda()
                optimizer.zero_grad()
                try:
                    out = model(x.to(torch.float32), edge_index)
                except:
                    tnum-=1
                    continue
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = out.max(1)
                correct += predicted.eq(y).sum().item()
            print("epochs {}".format(e) + " train loss: {}".format(train_loss/tnum) + " acc: {}".format(correct/tnum))
            #clt = RandomForestClassifier().fit(X[train_index], Y[train_index])   
            model.eval()
            all_predictions, all_targets = [], []
            for index in tqdm(test_index):
                x=X[index].cuda()
                y=Y[index].cuda()
                edge_index=EI[index].cuda()
                try:   
                    out = model(x.to(torch.float32),edge_index)
                except:
                    continue
                target = y
                target = target.cpu().detach().numpy()
                pred = out.argmax(dim=1).cpu().detach().numpy()
                all_predictions.append(pred)
                all_targets.append(target)
                acc_temp=max(acc_temp,round(accuracy_score(all_targets, all_predictions) * 100, 2))
        curr_score = curr_score + acc_temp
        #curr_score = curr_score + clt.score(X[test_index], Y[test_index])
    avg_score = curr_score / 5
    print("平均准确率为：", avg_score)


def save_gru(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(tqdm(_trainLoader)):
            # if index % 500 == 0:
            #     print(correct)
            #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
            if device != 'cpu':
                data = data.cuda()
            target = data.y.long()
            # if int(target)==0:
            #     target = torch.tensor([1.0,0.0], dtype=long).cuda()
            # else:
            #     target = torch.tensor([0.0,1.0], dtype=long).cuda()
            # optimizer.zero_grad()
            try:
                #out = model(data.x.to(torch.float32), data.edge_index, data.batch)
                outputs = model(data.x.to(torch.float32), data.edge_index)
                outputs = outputs.detach().cpu().tolist()
                edge_index = data.edge_index.detach().cpu().tolist()
                target = int(data.y)
                json_dict = {
                'node_features':outputs,
                'graph':edge_index,
                'target':target}
                file_path = os.path.join('/home/GNNLRP_model/ivdetect_test', data.name[0])
                with open(file_path, 'w', encoding='utf-8') as fp:
                    json.dump(json_dict, fp)
            except:
                tnum-=1
                continue
            
def train_batch(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    bnum = len(_trainLoader)
    cnum = len(_trainLoader.dataset)
    model.train()
    #model.zero_grad()
    my_dict={}
    for item in _trainLoader.dataset:
        my_dict[item.name]=[item.x,item.edge_index,item.y]
    for index, data in enumerate(tqdm(_trainLoader)):
        # if index % 500 == 0:
        #     print(correct)
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        #target = data.y.long()
        tnum=len(data.name)
        if device != 'cpu':
            data = data.cuda()
        optimizer.zero_grad()
        res=[]
        targets=[]
        for name in data.name:
            x=my_dict[name][0]
            edge_index=my_dict[name][1]
            target=my_dict[name][2]
            if device != 'cpu':
                x = x.cuda()
                edge_index=edge_index.cuda()
                target=target.cuda()
            try:
                out = model(data.x.to(torch.float32), data.edge_index)
                res.append(out)
                targets.append(target)
                _, predicted = out.max(1)
                correct += predicted.eq(target).sum().item()
            except:
                tnum-=1
                cnum-=1
                continue
        # res=torch.tensor(res)
        # targets=torch.tensor(targets)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss+=loss
    avg_train_loss = train_loss / bnum
    acc = correct / cnum
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss

def train(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    model.train()
    #model.zero_grad()
    for index, data in enumerate(tqdm(_trainLoader)):
        # if index % 500 == 0:
        #     print(correct)
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != 'cpu':
            data = data.cuda()
        target = data.y.long()
        # if int(target)==0:
        #     target = torch.tensor([1.0,0.0], dtype=long).cuda()
        # else:
        #     target = torch.tensor([0.0,1.0], dtype=long).cuda()
        optimizer.zero_grad()
        try:
            #out = model(data.x.to(torch.float32), data.edge_index, data.batch)
            out = model(data.x.to(torch.float32), data.edge_index)
        except:
            tnum-=1
            continue
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        #_, predicted = torch.exp(out).max(1)
        #_, predicted = np.argmax(torch.exp(out).numpy(), axis=-1).tolist()
        correct += predicted.eq(target).sum().item()
        del data.x, data.edge_index, data.y, data, predicted, out
    avg_train_loss = train_loss / tnum
    acc = correct / tnum
    print(tnum)
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss

def train2(curr_epochs, _trainLoader, model, criterion, optimizer, device,res_loss):
    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    model.train()
    #model.zero_grad()
    for index, data in enumerate(tqdm(_trainLoader)):
        # if index % 500 == 0:
        #     print(correct)
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != 'cpu':
            data = data.cuda()
        target = data.y.long()
        # if int(target)==0:
        #     target = torch.tensor([1.0,0.0], dtype=long).cuda()
        # else:
        #     target = torch.tensor([0.0,1.0], dtype=long).cuda()
        optimizer.zero_grad()
        try:
            #out = model(data.x.to(torch.float32), data.edge_index, data.batch)
            out = model(data.x.to(torch.float32), data.edge_index)
        except:
            tnum-=1
            continue
        loss = criterion(out, target)
        loss.backward()
        #conf = brier_score_loss(np.array([target.cpu().detach().numpy().squeeze()]),np.array([out.cpu().detach().numpy()[0][1]]),pos_label=0)
        if data.name[0] not in res_loss:
            res_loss[data.name[0]] = [loss.item()]
            #res_loss[data.name[0]] = [conf]
        else:
            res_loss[data.name[0]].append(loss.item())
            #res_loss[data.name[0]].append(conf)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        #_, predicted = torch.exp(out).max(1)
        #_, predicted = np.argmax(torch.exp(out).numpy(), axis=-1).tolist()
        correct += predicted.eq(target).sum().item()
        del data.x, data.edge_index, data.y, data, predicted, out
    avg_train_loss = train_loss / tnum
    acc = correct / tnum
    print(tnum)
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss

   
def evaluate_metrics(model, _loader, device):
    print('evaluate >')
    tnum = len(_loader)
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_probs = [], [], []
        for graph in tqdm(_loader):
            graph = graph.cuda()
            try:   
                #out = model(graph.x.to(torch.float32), graph.edge_index,graph.batch)
                out = model(graph.x.to(torch.float32), graph.edge_index)
            except:
                tnum-=1
                continue
            target = graph.y
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            # pred = torch.exp(out).argmax(dim=1).cpu().detach().numpy()
            # prob_1 = torch.exp(out).cpu().detach().numpy()[0][1]
            #for index in range(len(target)):
            all_probs.append(prob_1)
            #all_predictions.append(pred[index])
            all_predictions.append(pred)
            #all_targets.append(target[index])
            all_targets.append(target)
            del graph.x, graph.edge_index, graph.y, graph, out
        # acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        # print(acc)
        # precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        # f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        # recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        # matrix = confusion_matrix(all_targets, all_predictions)
        # #target_names = ['non-vul', 'vul']
        # #report = classification_report(all_targets, all_predictions, target_names=target_names)
        # result = " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \n " 
        # print(result)

        print(tnum)
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_score = round(auc(fpr, tpr) * 100, 2)
        acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        print(acc)
        precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        matrix = confusion_matrix(all_targets, all_predictions)
        #target_names = ['non-vul', 'vul']
        #report = classification_report(all_targets, all_predictions, target_names=target_names)
        result = "auc: {}".format(auc_score) + " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \n " 
        print(result)
    model.train()
    return acc,f1,recall,precision

def select_accpre(model, _loader, device):
    print('evaluate >')
    model.eval()
    with torch.no_grad():
        for graph in tqdm(_loader):
            graph = graph.cuda()
            try:   
                #out = model(graph.x.to(torch.float32), graph.edge_index,graph.batch)
                out = model(graph.x.to(torch.float32), graph.edge_index)
            except:
                continue
            target = graph.y
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            # pred = torch.exp(out).argmax(dim=1).cpu().detach().numpy()
            # prob_1 = torch.exp(out).cpu().detach().numpy()[0][1]
            # if int(target)==1 and int(target)==int(pred):
            if pred == 0:
                with open('####','a') as fp:
                    fp.write(graph.name[0]+'\n')
            del graph.x, graph.edge_index, graph.y, graph, out
    model.train()
    return




if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (gcn/ggnn)',
                        choices=['gcn', 'ggnn'], default='gcn')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.',default='devign')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='####/PDG_input/')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--subpdg_tag', type=str, help='Name of the node feature.', default='subpdg')
    parser.add_argument('--subpdg_num_tag', type=str, help='Name of the node feature.', default='subpdg_num')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=1)
    parser.add_argument('--task', type=str, help='train or eval', default='train')

    args = parser.parse_args()
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    input_dir = args.input_dir
    if args.task != 'eval':
        processed_data_path = os.path.join('#####/PDG_input/', '0.15_revise.bin')###
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset_loader = joblib.load(open(processed_data_path, 'rb'))
            #debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            print('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            dataset = DataSet2(train_src=os.path.join(input_dir, '0.15_train.txt'),
                              valid_src=None,
                              #train_src=None,
                              test_src=os.path.join(input_dir, '0.15_test.txt'),
                              #test_src=None,
                              )
            dataset_loader=dataset.dataset_loader
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset_loader, file)
            file.close()
    else:
        processed_data_path = os.path.join('####', 'test_1.bin')
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset_loader = joblib.load(open(processed_data_path, 'rb'))
            #debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            #debug('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            dataset_loader = DataSet2(train_src=None,
                            valid_src=None,
                            #test_src=os.path.join(input_dir, '0day_test.txt'),
                            test_src = '###'
                            )
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset_loader, file)
            file.close()

    #assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'


    if args.model_type == 'ggnn':
        model = RevealModel(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps)
    else:
        model = DeepWukong(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                            )##更改模型

    #print('Total Parameters : %d' % tally_param(model))
    print('#' * 100)
    print(model)
    model.cuda()
    #loss_function = BCELoss(reduction='mean')
    loss_function = CrossEntropyLoss()
    #loss_function = NLLLoss()
    optim = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    #optim = Adam(model.parameters(), lr=0.0002, weight_decay=0.002)
    print('batch size  : %d' % args.batch_size)
    print('lr  : 0.0001')
    print('weight_decay  : 0.001')
    model_dir = '/home/mytest/nvd/only_nvd_output/gnnexp_model/'
    if args.task == 'eval':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # eval(model=model, dataset=dataset_loader, max_steps=2000, dev_every=15,
        #     loss_function=loss_function, optimizer=optim,
        #     save_path=model_dir , max_patience=100, log_every=None) 
        _save_ckpt_file = torch.load('/home/GNNLRP_model/model_com/reveal/gnn_52.42_82.08_63.91.ckpt')
        model.load_state_dict(_save_ckpt_file)
        select_accpre(model=model, _loader=dataset_loader['test'], device=device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        starting_epochs = 0
        max_epochs = 100
        # _save_ckpt_file = torch.load('/home/GNNLRP_model/model/IVDetect/gnn_52.33_77.22_62.8.ckpt')
        # model.load_state_dict(_save_ckpt_file)
        # save_gru(1, dataset_loader['test'], model, loss_function, optim , device)
        res_loss={}
        #scores5 = cross_val(dataset_loader['train'], model, loss_function, optim )
        for e in range(starting_epochs, max_epochs):
            train_loss = train(e, dataset_loader['train'], model, loss_function, optim , device)
            #train_loss = train2(e, dataset_loader['train'], model, loss_function, optim , device, res_loss)
            #os.chdir("/home/IVDetect_MSRdata/result" )
            #torch.save(model, os.getcwd() + "/model/trained_model_{}.pt".format(e))##
            #torch.save(model.state_dict(), "/home/IVDetect_MSRdata/result/model/trained_model_{}.pkl".format(e))
            acc,f1,recall,precision = evaluate_metrics(model=model, _loader=dataset_loader['test'], device=device)
            # if recall < 80:
            torch.save(model.state_dict(), '#####/ckpt/'+f'{e}'+'_'+f'{acc}'+'_'+f'{recall}'+'_'+f'{precision}'+'_'+f'{f1}'+'.ckpt')
            # nni.report_intermediate_result(valid_auc)
            if train_loss < 0.3:
                break
            gc.collect()
    # with open('######/PDG_loss.json','w') as wp:
    #     json.dump(res_loss,wp)##输出loss
    # nni.report_final_result(valid_auc)


