import os 
import os.path as p 
import numpy as np 
import torch
import pickle
import random
from torch.utils.data import Dataset
import scipy.io as scio
import scipy.sparse as sp




def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def t_softmax(d,t=1):
    for i in range(len(d)):
        d[i] = d[i]*t
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d


def loadData(args,train=True):
    data_folder = p.join(args['src_path'],args['dataset'])
    if args['type'] == 'recovery':
        file_path = p.join(data_folder,args['dataset'])+'.plk'
        with open(file_path,'rb') as f:
            info = pickle.load(f)
            data = pickle.load(f)
    else:
        file_path = p.join(data_folder,args['dataset'])+str(args['split'])+'.plk'
        with open(file_path,'rb') as f:
            info = pickle.load(f)
            trainData = pickle.load(f)
            testData = pickle.load(f)
        if train:
            data = trainData
        else:
            data = testData
    return info,data


def loadAdj(args):
    data_folder = p.join(args['adj_path'], args['dataset'])
    if args['type'] == 'recovery':
        file_path = p.join(data_folder,args['dataset'])+'_adj.mat'
    else:
        file_path = p.join(data_folder,args['dataset'])+'_'+str(args['split'])+'_adj.mat'
    adj = scio.loadmat(file_path)
    return adj
    

def loadTest(args):
    target_folder = p.join(args['src_path'],args['dataset'])
    result_folder = p.join(args['dst_path'],args['dataset'])
    target_path = p.join(target_folder,args['dataset'])+'_d.plk'
    result_path = p.join(result_folder,args['dataset'])+'_LE.mat'
    with open(target_path, 'rb') as f:
        targets = pickle.load(f)
    preds = scio.loadmat(result_path)['distributions']
    return targets, preds


def mll_rec_loss(preds, targets, eps = 1e-12):
    w1 = 1 / targets.sum(1)
    loss = -targets*(torch.log(preds + eps))
    loss = loss.sum(1)*w1
    return loss.mean(0)


def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(sigma,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()


def preprocess_graph(adj, args):
    adj = sp.csr_matrix(adj)
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj + adj.T
    adj_label = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.sigmoid(torch.from_numpy(adj_label.todense()).to(torch.float32)).to(args['device']), torch.from_numpy(adj_norm.todense()).to(torch.float32).to(args['device'])


class MLLDataset(Dataset):
    def __init__(self, args):
        super(MLLDataset, self).__init__()
        info, datas = loadData(args)
        self.n_feature = info['n_feature']
        self.n_label = info['n_label']
        self.sparse = info['sparse']
        self.genDataSets(datas)

    def genDataSets(self,datas):
        dataSet = []
        n_sample = datas['length']
        feature_data = datas['data']
        label_data = datas['label']
        features = torch.zeros(n_sample,self.n_feature)
        labels = torch.zeros(n_sample,self.n_label)
        if self.sparse:
            for i in range(n_sample):
                feature = torch.from_numpy(np.array(feature_data[i],dtype=np.int64))
                label = torch.from_numpy(np.array(label_data[i],dtype=np.int64))
                if len(feature) > 0:
                    features[i].scatter_(0, feature,1)
                if len(label) > 0:
                    labels[i].scatter_(0,label,1)
        else:
            for i in range(n_sample):
                feature = torch.from_numpy(np.array(feature_data[i]))
                label = torch.from_numpy(np.array(label_data[i]))
                features[i] = feature
                labels[i] = label
            # Normalization
            max_data = features.max()
            min_data = features.min()
            features = (features-min_data) / (max_data - min_data)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index],self.labels[index]
