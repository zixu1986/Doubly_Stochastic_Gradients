function phix = rbffeature3_nofix(x, s, n, ri)
% random features for rbf kernel, cf. 
% Random Features for Large-Scale Kernel Machines
% Ali Rahimi and Benjamin Recht, NIPS 2007.
%
% x: original data, # of dimension x # of data points;
% s: scale parameter for rbf kernel exp(-s||x-x'||^2);
% n: the generated features are of dimension 2xn;
%
% Multi-dimensional code;

dimno = size(x, 1);

randn('seed', ri);

r = sqrt(2*s) * randn(dimno, n);

tmp = r'*x;

% Using cos and sin for now. Can also use cos with uniform random shift.
phix = [cos(tmp); sin(tmp)] ./ sqrt(n);
