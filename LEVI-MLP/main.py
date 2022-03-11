import torch.nn.functional as F 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import scipy.io as scio
import time

from model import *
from utils import * 
from test import calculate_dists


def train(enc, dec, optimizer, scheduler, features, labels, args):
    device = args['device']
    enc = enc.to(device)
    dec = dec.to(device)
    enc.train()
    dec.train()
    # records
    train_loss = []
    train_recx_loss = []
    train_recy_loss = []
    train_kl_loss = []

    features = features.to(torch.float32).to(device)
    labels = labels.to(torch.float32).to(device)
    input_data = torch.cat((features, labels),1)

    minloss = 1e10
    
    for epoch in range(args['epochs']):
        scheduler.step()
        t = time.time()
        # forward
        (mu, sigma) = enc(input_data)
        z = mu + sigma * (torch.randn(mu.size()).to(device))
        x_hat, y_hat = dec(z)
        # loss
        if args['type'] == 'classification' and args['data_type'] == 'binary':
            rec_loss_x = F.binary_cross_entropy(x_hat, features)
        else:
            rec_loss_x = F.mse_loss(x_hat, features)
        kl_loss = gauss_kl_loss(mu, sigma)
        rec_loss_y = F.binary_cross_entropy(y_hat, labels)
        loss = args['ld'] * rec_loss_y + args['alpha'] * kl_loss + args['beta'] * rec_loss_x
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        train_loss.append(loss.data.cpu())
        train_recx_loss.append(rec_loss_x.data.cpu())
        train_kl_loss.append(kl_loss.data.cpu())
        train_recy_loss.append(rec_loss_y.data.cpu())
        # print
        if epoch % 100 == 99:
            print('Epoch {:04d}: '.format(epoch + 1))
            print('loss: {:.03f} '.format(np.mean(train_loss)), 
                'kl_loss: {:.03f} '.format(np.mean(train_kl_loss)),
                'recx_loss: {:.03f} '.format(np.mean(train_recx_loss)),
                'recy_loss: {:.03f} '.format(np.mean(train_recy_loss)),
                'time: {:.5f}'.format(time.time()-t)
            )
        if np.mean(train_loss) < minloss:
            minloss = np.mean(train_loss)
    # print("minloss: ", minloss)

def label_enhance(enc, features, labels, n_instance, n_label, args):
    device = args['device']
    enc.to(device)
    enc.eval()
    indices = np.arange(n_instance)

    distributions = []
    batch_data = torch.cat((features, labels),1).to(torch.float32).to(device)
    # forward
    (mu, sigma) = enc(batch_data)
    d = F.sigmoid(mu[:, -n_label:])
    distributions.extend(d.data.cpu().numpy())
    return distributions

def save_data(distribution, args):
    if not p.isdir(args['dst_path']):
        os.mkdir(args['dst_path'])
    data_folder = p.join(args['dst_path'], args['dataset'])
    if not p.isdir(data_folder):
        os.mkdir(data_folder)
    if args['type'] == 'recovery':
        dst_path = p.join(data_folder,args['dataset'])+'_LE.mat'
    else:
        dst_path = p.join(data_folder,args['dataset'])+'_LE'+str(args['split'])+'_epo' + str(args['epochs']) + '.mat'
    distribution = np.array(distribution, dtype = np.float64)

    if args['type'] == 'recovery':
        mat_data ={ 'distributions':t_softmax(distribution)}
    else:
        mat_data = {'train_distributions': distribution}
    scio.savemat(dst_path, mat_data)

def main(args):
    device = torch.device('cuda:'+str(args['gpu']) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    args['device'] = device
    setup_seed(args['seed'])
    # create data
    train_data = MLLDataset(args)
    features = train_data.features
    labels = train_data.labels
    n_instance = train_data.features.shape[0]
    n_feature = train_data.n_feature
    n_label = train_data.n_label
    args['dim_z'] = max(args['dim_z'], n_label)
    # create model
    
    enc = VAE_Encoder(n_in=n_feature+n_label, n_hidden=args['n_hidden'],n_out=args['dim_z'],keep_prob=args['keep_prob'])
    dec = VAE_Decoder(n_in=args['dim_z'], n_hid=args['n_hidden'], n_out=n_feature, n_label=n_label, keep_prob=args['keep_prob'])
    optimizer = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=args['learning_rate'],weight_decay=1e-5)
    if args['type'] == 'recovery':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500,800,900], gamma=0.2, last_epoch=-1)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[300,400,500], gamma=0.2, last_epoch=-1)
    # training
    print('Begin training pharse')
    train(enc, dec, optimizer, scheduler, features, labels, args)
    # enhance label
    distribution = label_enhance(enc, features, labels, n_instance, n_label, args)
    # save distributions
    save_data(distribution, args)
    
    if args['type'] == 'recovery':
        targets, preds = loadTest(args)
        dists = calculate_dists(preds, targets)
        print(np.round(dists, 3))

