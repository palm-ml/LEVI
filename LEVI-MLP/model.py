import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


class VAE_Encoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Encoder,self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out*2)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
    
    def forward(self,inputs,eps=1e-8):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = self.fc_out(h1)
        mean = out[:,:self.n_out]
        std = F.softplus(out[:,self.n_out:]) + eps
        return (mean,std)

class VAE_Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(VAE_Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = self.fc_out(h1)
        labels_hat = F.sigmoid(z[:, -self.n_label:])
        return features_hat, labels_hat

class VAE_Bernulli_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Bernulli_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = F.sigmoid(self.fc_out(h1))
        return out

class VAE_Gauss_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Gauss_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_mean = nn.Linear(n_hidden,n_out)
        self.fc_var = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        mean = self.fc_mean(h1)
        var = F.softplus(self.fc_var(h1))
        return mean,var


class VGAE_Encoder(nn.Module):
    def __init__(self, adj, n_in, n_hid, n_out):
        super(VGAE_Encoder,self).__init__()
        self.n_out = n_out
        self.base_gcn = GraphConv(n_in, n_hid, adj)
        self.gcn = GraphConv(n_hid, n_out*2, adj, activation=lambda x:x)

    def forward(self, x):
        hidden = self.base_gcn(x)
        out = self.gcn(hidden)
        mean = out[:,:self.n_out]
        std = out[:,self.n_out:]
        return (mean,std)


class VGAE_Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(VGAE_Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = F.sigmoid(self.fc_out(h1))
        labels_hat = F.sigmoid(z[:, -self.n_label:])
        adj_hat = dot_product_decode(z)
        return features_hat, labels_hat, adj_hat


class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, adj, activation = F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)
