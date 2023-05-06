from operator import le, truediv
import pickle
import os
import json
import hashlib
from turtle import position
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import warnings
import csv
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

class KMeanClassifier():
    """默认使用欧式距离"""
    def __init__(self, X_train: np.asarray, y_train: np.asarray,
                  savefile="###"):
        self.X_train = X_train
        self.y_train = y_train
        self.savefile = savefile
        if not os.path.exists(savefile):
            self.data = self.__calClassCenter()
        self.data = pickle.load(open(self.savefile,"rb"))

    # 2.训练样本按标签聚类，计算每个类的中心
    def __calClassCenter(self):
        # 按类别建立一个dict
        dataset={}
        for x,y in zip(self.X_train,self.y_train):
            if y not in dataset:
                dataset[y]=[]
            dataset[y].append(x)

        # 计算每个类别的中心
        data = {}
        center = []
        labels = []
        for label in dataset:
            # data[label]=np.mean(np.asarray(dataset[label]),0)
            labels.append(label)
            center.append(np.mean(np.asarray(dataset[label]),0))
            # center.append(np.median(np.asarray(dataset[label]),0))

        data["label"] = labels
        data["center"] = center
        data['X_t'] = self.X_train
        data['Y_t'] = self.y_train

        # 将这个dict保存，下次就可以不用再重新建立(节省时间)
        pickle.dump(data,open(self.savefile,"wb"))
        #return data

    # 3.预测样本
    def predict(self,X_test: np.asarray)->np.asarray:
        labels = np.asarray(self.data["label"])
        center = np.asarray(self.data["center"])
        result_dist = np.zeros([len(X_test), len(center)])
        for i, data in enumerate(X_test):
            data = np.tile(data, (len(center), 1))
            distance = np.sqrt(np.sum((data - center) ** 2, -1))
            result_dist[i] = distance

        # 距离从小到大排序获取索引
        result_index = np.argsort(result_dist, -1)

        # 将索引替换成对应的标签，取距离最小对应的类别
        y_pred = labels[result_index][...,0]

        return y_pred

    # 4.计算精度信息
    def accuracy(self,y_true,y_pred)->float:
        return round(np.sum(y_pred == y_true) / len(y_pred),5)


if __name__ == '__main__':
    cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
    cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    with open('###','r') as f1:
        no_revise_path=f1.readlines()
    X_t=[]
    Y_t=[]
    for filename in tqdm(no_revise_path):
        try:
            with open(filename.strip(),'r') as f:
                file = json.load(f)
        except:
            filename2=filename.strip().split('/')[-1]
            if int(filename2[0]) == 0:
                filename2='1_CVE'+filename2.split('CVE')[-1]
                filename=filename.strip().split('0\'_CVE')[0]+filename2
                with open(filename,'r') as f:
                    file = json.load(f)
                file['target']=0
            else:
                filename2='0_CVE'+filename2.split('CVE')[-1]
                filename=filename.strip().split('1\'_CVE')[0]+filename2
                with open(filename,'r') as f:
                    file = json.load(f)
                file['target']=1
        slice_emb = np.array(file['node_features'])
        #pca = PCA(n_components=2)
        try:
            reduced = TSNE(n_components=2,random_state=0,init='pca',perplexity=5).fit_transform(slice_emb)
        except:
            continue
        #reduced = pca.fit_transform(slice_emb)
        t = reduced.transpose()
        new_emb=np.array(np.mean(t, axis=1))
        X_t.append([new_emb[0],new_emb[1]])
        Y_t.append(file['target'])
    
    clf = KMeanClassifier(X_t,Y_t)
    #plt.subplot(231)
    X_t=clf.data['X_t']
    Y_t=clf.data['Y_t']
    X_t=np.array(X_t)
    Y_t=np.array(Y_t)
    plt.scatter(X_t[:, 0], X_t[:, 1], c=Y_t, s=0.5, cmap=cm,edgecolors='none')
    #plt.title(u'reveal')
    plt.scatter(np.array(clf.data['center'])[:,0], np.array(clf.data['center'])[:,1],c=range(2), s=1, cmap=cm2,edgecolors='none')
    plt.xticks(())
    plt.yticks(())
    plt.savefig("###",dpi=1000,bbox_inches = 'tight')
    dis=(((clf.data['center'][0][0]-clf.data['center'][1][0])**2+(clf.data['center'][0][1]-clf.data['center'][1][1])**2)**0.5)
    print(dis)