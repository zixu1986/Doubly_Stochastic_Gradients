% Preprocess data.

% This should be the same as your output path pattern in
% transform_8m_dataset.m.
train_datapath_pattern = '/nv/hcoc1/bxie33/data/mnist8m_dataset/data_batch_%i.mat';
% if strcmp(train_datapath_pattern, '/nv/hcoc1/bxie33/data/mnist8m_dataset/data_batch_%i.mat')
%     error('Modify train_datapath_pattern to point to Matlab file batches!');
% end

test_datapath = sprintf(train_datapath_pattern, 82);
if ~exist('testlabel', 'var') || ~exist('testdata', 'var')
    load(test_datapath);
    testdata = data;
    testlabel = label;
end

n_lines = 8100000;
n_dim = 784;

batch_size = 100000;
n_batches = n_lines / batch_size;

if load_all
    traindata = zeros(n_dim, n_lines, 'single');
    % Serious index overflow was caused by using single precision.
    trainlabel = zeros(n_lines, 1, 'double');
    for i = 1:n_batches
        fprintf('processing batch %i\n', i);
        input_file = sprintf(train_datapath_pattern, i);
        tmp_f = load(input_file);
        d_idx = (i-1)*batch_size+1:i*batch_size;
        traindata(:, d_idx) = tmp_f.data;
        trainlabel(d_idx) = tmp_f.label;
    end
end

% Normalize each image to have unit L2 length.
normtrain = sqrt(sum(traindata.^2, 1)); 
normtest = sqrt(sum(testdata.^2, 1)); 

traindata = bsxfun(@rdivide, traindata, normtrain);
testdata = bsxfun(@rdivide, testdata, normtest);

ntr = size(traindata,2); 
nte = size(testdata, 2); 

k = length(unique(trainlabel));

trainY = zeros(k, ntr, 'single');
tl_idx = sub2ind([k, ntr], trainlabel+1, (1:ntr)');
trainY(tl_idx) = 1;
testY = zeros(k, nte, 'single');
tl_idx = sub2ind([k, nte], testlabel+1, (1:nte)');
testY(tl_idx) = 1;
 
% PCA on data. 
fprintf('-- pca of data ...\n');
subsample_size = 1e5;
subsample_idx = randsample(ntr, subsample_size);
covmat = traindata(:, subsample_idx) * traindata(:, subsample_idx)' ./ subsample_size; 
opts.isreal = true; 
pca_dim = 100; 
[v, ss] = eigs(double(covmat), pca_dim, 'LM', opts); 

traindata = v' * traindata; 
testdata = v' * testdata; 