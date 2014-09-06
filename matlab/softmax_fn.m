function p = softmax_fn(y)
% Nnumerically stable softmax.

max_y = max(y);
ny = exp(bsxfun(@minus, y, max_y));
p = bsxfun(@rdivide, ny, sum(ny));

if any(isnan(p))
    error('NaN in softmax!')
end