close all; clear; clc

load_all = true;
preprocess_8m_data

[~, train_true_y] = max(trainY,[],1);
[~, test_true_y] = max(testY,[],1);

% Find the median pairwise distance.
rand('seed', 1);
rperm = randperm(ntr); 
dismat = pdist(traindata(:,rperm(1:4000))'); 
s_coeff = 1;
s = 1 ./ (s_coeff * median(dismat)).^2 

reg_param = 0;
step_size0 = 1;
step_size1 = 1e-4;

show_error = 1; 

n = 2^20
blocksz = 4096
batch_size = 32768
blockno = fix(n / blocksz);
% Random seed offset.
r = 1

iters = blockno / 2;

train_error_mat(iters) = 0; 
test_error_mat(iters) = 0;

W = zeros(k, 2*n);

batch_idx = [1:batch_size];
test_preds = zeros(k, nte);

for j = 1:iters     
    fprintf('--iters no %d\n', j); 

    % Data already shuffled.
    batch_idx = mod(batch_idx + batch_size - 1, ntr) + 1;
    batch_data = traindata(:, batch_idx);
    f_idx = j - 1;
    testX = rbffeature3_nofix(testdata, s, blocksz, r*n+f_idx*blocksz);

    w_idx = f_idx*2*blocksz+1:(f_idx+1)*2*blocksz;
    train_batch_X = rbffeature3_nofix(batch_data, s, blocksz, r*n+f_idx*blocksz);

    % Accumulate residue.
    train_batch_preds = zeros(k, batch_size);
    for inner_j = 0:f_idx-1
        inner_w_idx = inner_j*2*blocksz+1:(inner_j+1)*2*blocksz;
        train_batch_preds = train_batch_preds + ...
            W(:, inner_w_idx) * rbffeature3_nofix(batch_data, s, blocksz, r*n+inner_j*blocksz);
    end
    residue = softmax_fn(train_batch_preds) - trainY(:, batch_idx);

    covx = train_batch_X * train_batch_X' / batch_size;
    preconditioner = covx + (reg_param + 1e-7) * eye(2*blocksz);

    step_size = step_size0 / (1 + step_size1 * j);
    updateW = - step_size * (residue * train_batch_X' / batch_size + reg_param * W(:, w_idx)) / preconditioner;
    W(:, w_idx) = W(:, w_idx) + updateW;
    if (reg_param > 1e-6)
        for inner_j = 0:f_idx-1
            inner_w_idx = inner_j*2*blocksz+1:(inner_j+1)*2*blocksz;
            W(:, inner_w_idx) = (1 - step_size * reg_param) * W(:, inner_w_idx);
        end
    end

    train_preds_batch = train_batch_preds + updateW * train_batch_X;

    [~, train_pred_y] = max(train_preds_batch, [], 1);
    train_error = sum(train_pred_y ~= train_true_y(batch_idx)) / batch_size;

    test_preds = test_preds + updateW * testX;
    [~, test_pred_y] = max(test_preds, [], 1);
    test_error = sum(test_pred_y ~= test_true_y) / nte;

    fprintf('---step size: %f\n', step_size)

    train_error_mat(j) = train_error; 
    fprintf('---train error: %f\n', train_error)

    test_error_mat(j) = test_error;
    fprintf('---test error: %f\n', test_error)
end
