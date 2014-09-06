% Transform the SVM data file to Matlab format.
clear; clc

% Modify this path to your downloaded, unzipped dataset file.
filename = '/nv/hcoc1/bxie33/data/mnist8m_dataset/mnist8m';
% Modify this to be the output file pattern.
output_pattern = '/nv/hcoc1/bxie33/data/mnist8m_dataset/mnist8m_%i.mat';

if strcmp(filename,'/nv/hcoc1/bxie33/data/mnist8m_dataset/mnist8m')
    error('Modify the mnist8m SVM data file!');
end
if strcmp(output_pattern, '/nv/hcoc1/bxie33/data/mnist8m_dataset/mnist8m_%i.mat')
    error('Modify the output file pattern!');
end

n_lines = 8100000;
n_dim = 784;

batch_size = 100000;
n_batches = n_lines / batch_size

fid = fopen(filename);

tline = fgetl(fid);
l_idx = 1;
b_idx = 1;
while ischar(tline)
    d_idx = mod(l_idx-1, batch_size)+1;
    if d_idx == 1
        traindata = zeros(n_dim, batch_size,'single');
        trainlabel = zeros(batch_size, 1, 'uint8');
    end
    
    [label, ~, ~, next_idx] = sscanf(tline, '%i', 1);
    feat = sscanf(tline(next_idx:end), '%i:%i');
    f_idx = feat(1:2:end);
    f_val = feat(2:2:end);

    trainlabel(d_idx) = label;
    traindata(f_idx, d_idx) = f_val / 255;
    
    if mod(l_idx, 5000) == 1
        fprintf('---processed %i / %i\n', l_idx, n_lines)
    end
    if d_idx == batch_size
        fprintf('Finished batch %i\n', b_idx)
        output_path = sprintf(output_pattern, b_idx);
        save(output_path, 'traindata', 'trainlabel')
        b_idx = b_idx + 1;
    end
    
    tline = fgetl(fid);
    l_idx = l_idx + 1;
end

fclose(fid);