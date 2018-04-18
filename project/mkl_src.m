%% MLK - SRC
clear all
close all
clc
addpath(genpath('toolbox_optim'));
%% Load some data
% Load images and labels
pos_point = [1 1 ]';
neg_point = [-2 -3 ]';

% Make 20 random points around pos and neg
num_samples=20;
Y = zeros(numel(pos_point), num_samples);
l = zeros(1, 10);
for i=1:num_samples/2
    Y(:,i) = awgn(pos_point, num_samples/2);
    l(i) = 1;
end
for i=num_samples/2 + 1:num_samples
    Y(:,i) = awgn(neg_point, num_samples/2);
    l(i) = 2;
end

for i=1:size(Y,2)
    Ynorm(:, i) = Y(:,i)/norm(Y(:,i));
end
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

% Make the ideal matrix - FIXME - assumes blocks of samples (probably fine)
K_ideal = eye(size(Y,2));
% Find the number of samples in each class
classes = unique(l);
num_classes = numel(classes);
masks = zeros(size(Y,2),numel(classes));
for i=1:num_classes
    num_samples_per_class(i) = sum(l == classes(i));
    masks(:,i) = l == classes(i);
    locs = find(l == classes(i));
    K_ideal(min(locs):max(locs),min(locs):max(locs)) = 1;
end
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
lambda = .01;
% max iterations
T = 10;
% error thresh for convergence
err_thresh = .01;
err = err_thresh + 1;

% total number of samples
% N = num_samples;
N = size(Y,2);
%% Initalize kernel weights
% Start by giving all weight to the most aligned kernel, i.e. ranked_mat{1}
eta = zeros(length(ranked_mats),1);
eta(1) = 1;
%% Find initial sparse coefficients matrix X
% For each ith training sample, 0 out the ith row
% of Y and call it Y_tilde, then solve for x_i in y_i=(Y_tilde)x_i
% by solving (18) in the paper.
% x_i = argmin_x (k(y_i, y_i)+ x^TK(Y_tilde, Y_tilde)x - 2K(y_i, Y_tilde)x - lambda||x_i||^1)
% FIXME? Uses Pseudo inverse to find initial x's
% Find sparse codes for each sample
x=zeros(size(Y,2), numel(l));
for i=1:size(Y,2)
    Yt = Ynorm;
%     Yt(:,i) = 0;
%     x(:,i) = getCoeffs(Y(:,i), Yt, lambda, 10000);
    x(:,i) = getCoeffs2(x(:,i), Y(:,i), Yt, lambda, eta, ranked_kappa, 5000);
end

%% Iterate until quitting conditions are satisfied
t=0;
h = zeros(1, size(Y,2));
while(t <= T && err>= err_thresh)
    % Compute K_m(Y_tilde, Y_tilde)

    % Precompute the predicted labels for each base kernel
    
    
    for i=1:N
        %         K = curr_kernel;
        %         K(i, :) = 0;
        %         K(:, i) = 0;
        Yt = Ynorm;
%         Yt(:,i) = 0;
        % Compute the sparse code x_i
        x(:,i) = getCoeffs2(x(:,i), Y(:,i), Yt, lambda, eta, ranked_kappa, 1);
        %         x(:,i) = getCoeffs(Y(:,i), Yt, lambda, 10000);
        % Compute the predicted label g_i using x_i
        h(i) = calcZis(eta, x(:,i), ranked_mats, Y(:,i), l);
        
    end
    
    g = zeros(length(kappa), N);
    for ker_num=1:length(kappa)
        for i=1:N
            g(ker_num, i) = calcZis(1, x(:,i), ranked_mats{ker_num}, Y(:,i), l);
        end
    end
    
    z = h == l;
    for m=1:length(kappa)
        zm(m,:) = g(m,:) == l;
        c(m,1) = sum(zm(m, find(z==0)))/sum(1-z);
    end
    % Update weights for all m
    % find the best new kernel based on c
    [best_c_val,best_c_idx] = max(c(c ~=0));
    all_non_0_c_vals = c(c ~=0);
    c_idxs = find(c~=0);
    non_0_c = c(c ~=0);
    for i=best_c_idx:-1:1
        if non_0_c(i) > best_c_val + mu
            best_c_idx = i;
            best_c_val = non_0_c(i);
        end
    end
    
    new_kernel = c_idxs(best_c_idx);
    new_kernel_weight = sum(bitand(zm(new_kernel,:), not(z)))/sum(bitor(not(z), not(zm(new_kernel,:))));
    curr_kernel_weight = sum(bitand(z, not(zm(new_kernel,:))))/sum(bitor(not(z), not(zm(new_kernel,:))));
    
    % Do the actual mixing weight update here
    for m=1:length(kappa)
        if m==new_kernel
            eta(m)=new_kernel_weight;
        else
            eta(m) = eta(m)*curr_kernel_weight;
        end
    end
    
    % Compute sums of all weights and normalize weights by sum
    total_weights = sum(eta);
    prev_eta = eta; % save for calcing error
    eta = eta/total_weights;
    
    % set err = ||eta^{t-1}-eta^{t}||_2
    err = norm(prev_eta - eta,2)
    
    t = t + 1;
end
curr_kernel = zeros(size(kernel_mats{1},1));
for i=1:length(eta)
    curr_kernel = curr_kernel + eta(i)*kernel_mats{i};
end