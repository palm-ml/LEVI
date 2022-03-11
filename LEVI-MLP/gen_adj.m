function adj = gen_adj(dataset)
    ktrail=10
    for i = 1:ktrail
        load(strcat('matdata/', dataset, '/', dataset, '_total_',num2str(i),'.mat'));
        w = estimate_w(train_data);
        adj = eye(size(w,1));
        mu = mean(w(:));
        sigma = std(w(:));
        adj(w>mu+sigma) = 1.0;
        save(strcat('adjmat/', dataset, '/', dataset, '_', num2str(i), '_adj.mat'), 'adj');
    end
end
