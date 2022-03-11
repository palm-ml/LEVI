datasets = {'Artificial', 'SJAFFE', 'Yeast_spoem', 'Yeast_spo5', 'Yeast_dtt', 'Yeast_cold', 'Yeast_heat', 'Yeast_spo', 'Yeast_diau', 'Yeast_elu', 'Yeast_cdc', 'Yeast_alpha', 'SBU_3DFE', 'Movie'};
K = 10;     % number of neighbours

for i = 1:length(datasets)
    dataset = datasets{i};
    load(strcat('matdata\', dataset, '\', dataset, '_binary.mat'));
    % [instanceNum, ~] = size(features);
    % adj = estimate_top_struct(features, 10);
    
    fprintf(1,'Estimate the Topological Structure.\n');

    [N,D] = size(features);

    neighborhood = knnsearch(features, features, 'K', K+1);
    neighborhood = neighborhood(:, 2:end);
    adj = zeros(N, N);
    
    for j=1:N
        neighbors = neighborhood(j,:);
        adj(j,neighbors) = ones(1,K);
    end
    save(strcat('adjmat\', dataset, '_adj.mat'), 'adj');
end