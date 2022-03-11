clc
clear
addpath(genpath(pwd));
% dataset_list
datasets = {'CAL500/CAL500','emotions/emotions','medical/medical','llog/llog','enron/enron',...
    'image/image','scene/scene','yeast/yeast','slashdot/slashdot','corel5k/corel5k',...
    'msra/msra','rcv1subset1/rcv1subset1','bibtex/bibtex'};
% folder of data
data_folder = '../matdata/';
% folder of distribution
distribution_folder = '../results/';

for dataN = 1:1
    dataset = strsplit(datasets{dataN},'/');
    dataset = dataset{1};
    data_path = strcat(data_folder, dataset, '/', dataset, '_total_');
    epos = [200];
    list1 = [0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 5, 7.5, 10, 50, 100];
    % list2 = [0.001, 0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5];
    list1 = [0.06,0.07,0.08,0.09,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45];
    % list1 = [0.105, 0.11, 0.115, 0.12];
    % list1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
    % list1 = [0.325, 0.350, 0.375, 0.425, 0.450, 0.475];
    % list2 = [0.81,0.82,0.83,0.84,0.86,0.87,0.88,0.89];
    % epos = [500]; 
    list1 = [0.4,0.6,0.8,1,1.2,1.4,1.6];
    list1 = [1];
    list2 = [0.01];
    list2 = [0.01,0.05,0.1,0.5,1,5,10];
    % epos = [10,20,30,50,100];
    % epos = [300,325,375,400,425,450,475,500,525,550,575,600,625,650,675,700,750,800,900];
    % epos = [5000];
    for C1 = 1:length(list1)
    	for C2 = 1:length(list2) 
            for epoid = 1:length(epos)
                epo = epos(epoid);
                
    distribution_path = strcat(distribution_folder, dataset, '/', dataset, '_LE');
    % parameter
    para.tol = 1e-5;    %tolerance during the iteration
    para.epsi = 0.001;  %instances whose distance computed is more than epsi should be penalized
    para.C1 = 2.08;        %penalty parameter
    para.C2 = 0.85;        %penalty parameter
    para.C1 = list1(C1);
    para.C2 = list2(C2);
    para.ker = 'rbf';   %type of kernel function ('lin', 'poly', 'rbf', 'sam')
    % variable to store measurement results
    dists = zeros(10,5);
    for group = 1:10
       % load distribution [n_sample, n_label]
       distribution_file = strcat(distribution_path, int2str(group),'_epo', num2str(epo), '.mat');
       % distribution_file = strcat(distribution_path, int2str(group), '.mat');
       load(distribution_file);
%        train_distributions = train_distributions;
       % load data
       data_file = strcat(data_path, int2str(group),'.mat');
       load(data_file);
       % preprocessing
       train_features = zscore(train_data); %[n_sample x n_feature]
       train_target = train_distributions;
       train_target = 1 ./ (1 + exp(0 - train_target)); % sigmoid(d_i^j)
       tmp_max = max(train_target,[],2);
       tmp_min = min(train_target,[],2);
       train_target = (train_target-tmp_min) ./ (tmp_max-tmp_min)*2 -1;
       test_target = test_target';
       test_target(find(test_target==0))=-1;
       para.par  = 1*mean(pdist(train_data)); %parameter of kernel function
       % training
       model = amsvr(train_data, train_target, para);
       % predicting
       [label, degree] = predict(test_data, train_data, model);
       % evaluation
       dist = testModel(degree, label, test_target);
       dists(group,:) = dist;
    end
    dist_mean = mean(dists,1);
    dist_std = std(dists,1);
    % fprintf(1, strcat("epos: ", num2str(epo)));
    fprintf(1, strcat("C1: ", num2str(para.C1), " C2: ", num2str(para.C2), " epos: ", num2str(epo)));
    round(dist_mean,3)
    round(dist_std,3)
    
            end
    	end     %C2
    end     %C1
end