function [x] = getCoeffs2(prev_x, y,A, lambda, eta, kappa, num_iter)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

full_kernel = zeros(size(A,2), size(A,2));
single_kappa = 0;
partial_kappa = zeros(1, size(A,2));
for m=1:length(kappa)
    K = zeros(size(A,2), size(A,2));
    for i=1:size(A,2)
        for j=1:size(A,2)
            K(i,j) = kappa{m}(A(:,i), A(:,j));
        end
        partial_kappa(i) = partial_kappa(i) +  eta(m)*kappa{m}(y, A(:,i));
    end
    full_kernel = full_kernel + eta(m)*K;
    single_kappa = single_kappa + eta(m)*kappa{m}(y,y);
end

% Dimension of the problem.
n = size(A,2);
p = size(A,1);
% List of benchmarked algorithms.
% methods = {'fb', 'fista', 'nesterov'};
methods = {'fista'};
% operator callbacks
F = @(x)lambda*norm(x,1);
G = @(x)single_kappa + x'*full_kernel*x-2*partial_kappa*x;
% Proximal operator of F.
ProxF = @(x,tau)perform_soft_thresholding(x,lambda*tau);
% Gradient operator of G.
GradG = @(x)2*full_kernel*x - 2*partial_kappa';
% Lipschitz constant.
L = norm(A)^2;
% Function to record the energy.
options.report = @(x)F(x)+G(x);
options.verb = 0;

% Bench the algorithm
options.niter = num_iter;
E = [];
for i=1:length(methods)
    options.method = methods{i};
    [x,e] = perform_fb(prev_x, ProxF, GradG, L, options);
    E(:,i) = e(:);
end
end

