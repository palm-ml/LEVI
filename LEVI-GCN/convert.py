import os
import os.path as p
import pickle

import numpy as np
import scipy.io as scio
from run_main import dataset_type, dataset_list


SRC_PATH = p.join(p.dirname(__file__),'matdata')
DST_PATH = p.join(p.dirname(__file__),'datasets')

def rec_convert(src_path, dst_path1, dst_path2):
    datas = scio.loadmat(src_path)
    features = np.array(datas['features'], dtype=np.float32)
    lds = np.array(datas['labelDistribution'], dtype=np.float32)
    labels = np.array(datas['logicalLabel'], dtype=np.int32)
    print(features.shape)
    print(lds.shape)
    print(labels.shape)
    train_data = {'data':features,
                  'label':labels,
                  'length':len(features)}
    info = {'n_feature':features.shape[1],
            'n_label':labels.shape[1],
            'sparse':False}
    with open(dst_path1,'wb') as f:
        pickle.dump(info, f)
        pickle.dump(train_data, f)
    print("save file %s" % dst_path1)
    with open(dst_path2,'wb') as f:
        pickle.dump(lds, f)
    print("save file %s" % dst_path2)


def load_cls_mat(src_path):
    datas = scio.loadmat(src_path)
    train_data = np.array(datas['train_data'], dtype=np.float32)
    train_labels = np.array(datas['train_target'], dtype = np.float32)
    test_data = np.array(datas['test_data'], dtype = np.float32)
    test_labels = np.array(datas['test_target'], dtype=np.float32)
    train_labels = (train_labels + 1) / 2
    train_labels = train_labels.astype(np.int32)
    test_labels = (test_labels + 1) / 2
    test_labels = test_labels.astype(np.int32)
    return train_data, train_labels, test_data, test_labels

def save_cls_data(dst_path, datas_train, labels_train, datas_test, labels_test):
    labels_train = labels_train.T
    labels_test = labels_test.T
    n_samples = len(datas_train)
    train_data = {'data':datas_train,
                  'label':labels_train,
                  'length':len(datas_train)}
    test_data = {'data':datas_test,
                 'label':labels_test,
                 'length':len(datas_test)}
    info = {'n_feature':datas_train.shape[1],
            'n_label':labels_train.shape[1],
            'sparse':False}
    with open(dst_path,'wb') as f:
        pickle.dump(info, f)
        pickle.dump(train_data, f)
        pickle.dump(test_data, f)
    print("save file %s" % dst_path)
    
if __name__ == '__main__':
    task_type = 'recovery'      # recovery or classification
    for idx in range(0, 1):     # dataset index
        dataset = dataset_list[task_type][idx]
        src_path = p.join(SRC_PATH,dataset)
        dst_path = p.join(DST_PATH,dataset)
        if not p.isdir(dst_path):
            os.mkdir(dst_path)
            
        if task_type == 'recovery':
            src_file = p.join(src_path, dataset+'_binary.mat')
            dst_file1 = p.join(dst_path, dataset+'.plk')
            dst_file2 = p.join(dst_path, dataset+'_d.plk')
            rec_convert(src_file, dst_file1, dst_file2)
        else:
            for i in range(10):
                src_file = p.join(src_path, dataset+'_total_'+str(i+1)+'.mat')
                dst_file = p.join(dst_path, dataset+str(i+1)+'.plk')
                datas_train, labels_train, datas_test, labels_test = load_cls_mat(src_file)
                save_cls_data(dst_file, datas_train, labels_train, datas_test, labels_test)
