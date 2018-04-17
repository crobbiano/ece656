%% MLK - SRC
clear all
close all
clc
%% Load some data
% Load images and labels
y1 = [1 2 1]'; l1 = 0;
y2 = [1.3 2.1 1.4]'; l2 = 0;
y3 = [.8 1.5 1]'; l3 = 0;
y4 = [-0.3 -0.7 -1.1]'; l4 = 1;
y5 = [-.5 -.8 -1]'; l5 = 1;
y6 = [-.3 -1.3 -1.3]'; l6 = 1;
% Build data sample matrix Y and label vector l
Y = [y1, y2, y3, y4, y5, y6];
l = [l1, l2, l3 ,l4, l5, l6];
%% Make kernel functions
% Choose the kernel functions and make vector of them
kappa  = { ...
    @(x,y) x'*y; ...            % Linear
    @(x,y) (x'*y + 1); ...
    @(x,y) (x'*y + 0.5)^2; ...  % Polynomial
    @(x,y) (x'*y + 0.5)^3; ...
    @(x,y) (x'*y + 0.5)^4; ...
    @(x,y) (x'*y + 1.0)^2; ...
    @(x,y) (x'*y + 1.0)^3; ...
    @(x,y) (x'*y + 1.0)^4; ...
    @(x,y) (x'*y + 1.5)^2; ...
    @(x,y) (x'*y + 1.5)^3; ...
    @(x,y) (x'*y + 1.5)^4; ...
    @(x,y) (x'*y + 2.0)^2; ...
    @(x,y) (x'*y + 2.0)^3; ...
    @(x,y) (x'*y + 2.0)^4; ...
    @(x,y) (x'*y + 2.5)^2; ...
    @(x,y) (x'*y + 2.5)^3; ...
    @(x,y) (x'*y + 2.5)^4; ...
    @(x,y) tanh(0.1 + 1.0*(x'*y)); ...  % Hyperbolic Tangent
    @(x,y) tanh(0.2 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.3 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.4 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 1.0*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.2*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.4*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.6*(x'*y)); ...
    @(x,y) tanh(0.5 + 0.8*(x'*y)); ...
    @(x,y) exp((-norm(x-y)^2/0.1)); ...  % Gaussian
    @(x,y) exp((-norm(x-y)^2/0.2)); ...
    @(x,y) exp((-norm(x-y)^2/0.3)); ...
    @(x,y) exp((-norm(x-y)^2/0.4)); ...
    @(x,y) exp((-norm(x-y)^2/0.5)); ...
    @(x,y) exp((-norm(x-y)^2/0.6)); ...
    @(x,y) exp((-norm(x-y)^2/0.7)); ...
    @(x,y) exp((-norm(x-y)^2/0.8)); ...
    @(x,y) exp((-norm(x-y)^2/0.9)); ...
    @(x,y) exp((-norm(x-y)^2/1.0)); ...
    @(x,y) exp((-norm(x-y)^2/1.1)); ...
    @(x,y) exp((-norm(x-y)^2/1.2)); ...
    @(x,y) exp((-norm(x-y)^2/1.3)); ...
    @(x,y) exp((-norm(x-y)^2/1.4)); ...
    @(x,y) exp((-norm(x-y)^2/1.5)); ...
    @(x,y) exp((-norm(x-y)^2/1.6)); ...
    @(x,y) exp((-norm(x-y)^2/1.7)); ...
    @(x,y) exp((-norm(x-y)^2/1.8)); ...
    };
%% Compute the M kernel matrices
% Find K_m(Y, Y) for all M kernel functions
kernel_mats = cell(length(kappa), 1);
for m=1:length(kappa)
    K = zeros(size(Y,2), size(Y,2));
    for i=1:size(Y,2)
        for j=1:size(Y,2)
            K(i,j) = kappa{m}(Y(:,i), Y(:,j));
        end
    end
    kernel_mats{m} = K;
end

% Make the ideal matrix - FIXME
K_ideal = eye(6);
K_ideal(1,2) = 1; K_ideal(2,1)=1;
K_ideal(1,3) = 1; K_ideal(3,1)=1;
K_ideal(2,3) = 1; K_ideal(3,2)=1;

K_ideal(4,5) = 1; K_ideal(5,4)=1;
K_ideal(5,6) = 1; K_ideal(6,5)=1;
K_ideal(6,4) = 1; K_ideal(4,6)=1;

%% Get ranked ordering of kernel matrices
ranked_mats = kernel_mats;
ranked_kappa = kappa;
for i=1:length(ranked_mats)
    alignment_scores(i) = kernelAlignment(kernel_mats{i}, K_ideal);
end
[sorted, idx] = sort(alignment_scores,'descend');
ranked_mats = ranked_mats(idx,:);
ranked_kappa = ranked_kappa(idx,:);
%% Setup other parameters
% sparsity_reg \mu
mu = .2;
% overfitting_reg \lambda
lambda = .1;
% max iterations
T = 100;
% error thresh for convergence
err_thresh = .5;
err = err_thresh + 1;

% total number of samples
% N = num_samples;
N = size(Y,2);

%% Find initial sparse coefficients matrix X
% For each ith training sample, 0 out the ith row
% of Y and call it Y_tilde, then solve for x_i in y_i=(Y_tilde)x_i
% by solving (18) in the paper.
% x_i = argmin_x (k(y_i, y_i)+ x^TK(Y_tilde, Y_tilde)x - 2K(y_i, Y_tilde)x - lambda||x_i||^1)
% FIXME? Uses Pseudo inverse to find initial x's
% Find sparse codes for each sample
x=zeros(size(Y,2), numel(l));
for i=1:size(Y,2)
    Yt = Y;
    Yt(:,i) = 0;
    x(:,i) = Yt'*inv(Yt*Yt')*Y(:,i);
end

%% Initalize kernel weights
% Start by giving all weight to the most aligned kernel, i.e. ranked_mat{1}
eta = zeros(1, length(ranked_mats));
eta(1) = 1;

%% Iterate until quitting conditions are satisfied
while(t <= T && err>= err_thresh)
    % Compute K_m(Y_tilde, Y_tilde)
    for i=1:N
        K = kernel_mats{m};
        K(i, :) = 0;
        K(:, i) = 0;
        % Compute the sparse code x_i as done before - MAKE FNC FOR THIS
        % Compute the predicted label g_i using x_i
    end
    % Update weights for all m
    
    % Compute sums of all weights and normalize weights by sum
    
    % set err = ||eta^{t-1}-eta^{t}||_2
    
end
