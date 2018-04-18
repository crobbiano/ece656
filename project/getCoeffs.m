function [x] = getCoeffs(y,A, lambda, num_iter)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Dimension of the problem.
n = size(A,2);
p = size(A,1);
% List of benchmarked algorithms.
% methods = {'fb', 'fista', 'nesterov'};
methods = {'fista'};
% operator callbacks
F = @(x)lambda*norm(x,1);
G = @(x)norm(y-A*x,2);
% Proximal operator of F.
ProxF = @(x,tau)perform_soft_thresholding(x,lambda*tau);
% Gradient operator of G.
GradG = @(x)A'*(A*x-y);
% Lipschitz constant.
L = norm(A)^2;
% Function to record the energy.
options.report = @(x)F(x)+G(x);

% Bench the algorithm
options.niter = num_iter;
E = [];
for i=1:length(methods)
    options.method = methods{i};
    [x,e] = perform_fb(zeros(n,1), ProxF, GradG, L, options);
    E(:,i) = e(:);
end
end

