%% Basis Pursuit Denoising with Forward-Backward
% Test the use of Forward-backward-like splitting for the resolution
% of a compressed sensing regularization.
clear all 
clc
addpath('../');
addpath('../toolbox/');



%%
% Regularization parameter.

lambda = .01;

%%
% Matrix and observation.

pos_point = [1 1 ]';
neg_point = [-2 -3 ]';

% Make 20 random points around pos and neg
Y = zeros(numel(pos_point), 40);
l = zeros(1, 40);
for i=1:20
    Y(:,i) = awgn(pos_point, 10);
    l(i) = 1;
end
for i=21:40
    Y(:,i) = awgn(neg_point, 10);
    l(i) = 2;
end 
y = Y(:,1);
for i=1:size(Y,2)
    Ynorm(:, i) = Y(:,i)/norm(Y(:,i));
end
A=Ynorm;
% A=Y;
A(:,1)=0;
%%
% Dimension of the problem.

n = size(Y,2);
p = size(Y,1);
%%
% List of benchmarked algorithms.

methods = {'fb', 'fista', 'nesterov'};
methods = {'nesterov'};

%%
% operator callbacks

F = @(x)lambda*norm(x,1);
G = @(x)norm(y-A*x,2);

%%
% Proximal operator of F. 

ProxF = @(x,tau)perform_soft_thresholding(x,lambda*tau);

%%
% Gradient operator of G.
GradG = @(x)A'*(A*x-y);

%%
% Lipschitz constant.

L = norm(A)^2;

%%
% Function to record the energy.

options.report = @(x)F(x)+G(x);

%%
% Bench the algorithm

options.niter = 10000;
E = [];
for i=1:length(methods)
    options.method = methods{i};
    [x,e] = perform_fb(zeros(n,1), ProxF, GradG, L, options);
%     [x2,e2] = perform_dr(zeros(n,1), ProxF, GradG, options);
    E(:,i) = e(:);
end
e = min(E(:));

%%
% Display the decays of the energies.

%%
% _IMPORTANT:_ Note that the comparison with Nesterov is unfair, since each
% Nesterov iteration cost twice more. 

sel = 1:round(options.niter/10);
loglog(E(sel,:)-e);
axis tight;
legend(methods);

A*x
y
norm(A*x-y,2)