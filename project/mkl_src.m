%% MLK - SRC
clear all
close all
clc
addpath(genpath('toolbox_optim'));
addpath(genpath('srv1_9'));
warning('off', 'MATLAB:nargchk:deprecated');
%% Load some data
% Load images and labels
pos_point = [30 30 40]';
neg_point = [29 31 37]';

% Make 20 random points around pos and neg
num_samples=50;
Y = zeros(numel(pos_point), num_samples);
l = zeros(1, 10);
for i=1:num_samples/2
    Y(:,i) = awgn(pos_point, 2);
    l(i) = 1;
end
for i=num_samples/2 + 1:num_samples
    Y(:,i) = awgn(neg_point, 2);
    l(i) = 2;
end

for i=1:size(Y,2)
    Ynorm(:, i) = Y(:,i)/norm(Y(:,i));
end
% Y = Ynorm;
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
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.1)); ...  % Gaussian
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.8)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/0.9)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.0)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.1)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.2)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.3)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.4)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.5)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.6)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.7)); ...
    @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/1.8)); ...
    };
kappa2  = { ...
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
    option.kernel = 'cust'; option.kernelfnc=kappa{m};
    kernel_mats{m} = computeKernelMatrix(Y,Y,option);
end

kernel_mats2 = cell(length(kappa), 1);
for m=1:length(kappa)
    K = zeros(size(Y,2), size(Y,2));
    for i=1:size(Y,2)
        for j=1:size(Y,2)
            K(i,j) = kappa2{m}(Y(:,i), Y(:,j));
        end
    end
    kernel_mats2{m} = K;
end

for i=1:length(kappa)
    errors(i) = sum(sum((kernel_mats{i}-kernel_mats2{i})^2));
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
% overfitting_reg \mu
mu = .2;
% sparsity_reg \lambda
lambda = .01;
% max iterations
T = 30;
% error thresh for convergence
err_thresh = .05;
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
% by solving (18) in the paper.
% x_i = argmin_x (k(y_i, y_i)+ x^TK(Y_tilde, Y_tilde)x - 2K(y_i, Y_tilde)x - lambda||x_i||^1)
x=zeros(size(Y,2), numel(l));
for i=1:size(Y,2)
    x(:,i) = getCoeffs(x(:,i), Y(:,i), Y, ranked_kappa, lambda, eta, 5000, i);
end

%% Iterate until quitting conditions are satisfied
t=0;
h = zeros(1, size(Y,2));
while(t <= T && err>= err_thresh)
%     x=zeros(size(Y,2), numel(l));
    for i=1:N
        % Compute the sparse code x_i
        x(:,i) = getCoeffs(x(:,i), Y(:,i), Y, ranked_kappa, lambda, eta, 5000, i);
        % Compute the predicted label h_i using x_i
        h(i) = calcZis(x(:,i), Y(:,i), Y, ranked_kappa, eta, l);
    end
    
    % Precompute the predicted labels for each base kernel
    g = zeros(length(kappa), N);
    for ker_num=1:length(kappa)
        for i=1:N
            g(ker_num, i) = calcZis(x(:,i), Y(:,i), Y, ranked_kappa{ker_num}, 1, l);
        end
    end
    
    z = (h == l);
    % If we have perfect guesses then quit
    if length(z)/sum(z) == 1
        err = 0;
    else
        for m=1:length(kappa)
            zm(m,:) = g(m,:) == l;
            c(m,1) = sum(zm(m, find(z==0)))/sum(1-z);
        end
        % Update weights for all m
        % find the best new kernel based on c
        [best_c_val,best_c_idx] = max(c(c ~=0 & eta == 0) );
        best_c_val_og = best_c_val;
        c_idxs = find(c~=0 & eta == 0);
        
        %     [best_c_val,best_c_idx] = max(c(c ~=0) );
        %     best_c_val_og = best_c_val;
        %     c_idxs = find(c~=0);
        
        best_c_idx = c_idxs(best_c_idx);
        for i=best_c_idx:-1:1
            if (c(i) ~=0 & eta == 0) & (c(i) + mu > best_c_val_og)
                %         if (c(i) ~=0) & (c(i) + mu > best_c_val_og)
                best_c_idx = i;
                best_c_val = c(i);
                display(['Changed best_c to higher index: ' num2str(i)])
            end
        end
        
        
        if isempty(best_c_idx)
            display(['Cheating'])
            new_kernel = 1;
            new_kernel_weight = .1;
            curr_kernel_weight = 1;
        else
            new_kernel = best_c_idx;
            new_kernel_weight = sum(bitand(zm(new_kernel,:), not(z)))/sum(bitor(not(z), not(zm(new_kernel,:))));
            curr_kernel_weight = sum(bitand(z, not(zm(new_kernel,:))))/sum(bitor(not(z), not(zm(new_kernel,:))));
        end
        
        prev_eta = eta; % save for calcing error
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
        eta = eta/total_weights;
        
        % set err = ||eta^{t-1}-eta^{t}||_2
        err = norm(prev_eta - eta,2);
    end
    t = t + 1;
    display(['Iteration: ' num2str(t) '/' num2str(T) ' Error: ' num2str(err)])
end

%% Classify some new points
pos_point2 = [30 30 40]';
neg_point2 = [29 31 37]';

% Make 20 random points around pos and neg
num_samples2=20;
Y2 = zeros(numel(pos_point2), num_samples2);
l2 = zeros(1, num_samples2);
for i=1:num_samples2/2
    Y2(:,i) = awgn(pos_point2, 2);
    l2(i) = 1;
end
for i=num_samples2/2 + 1:num_samples2
    Y2(:,i) = awgn(neg_point2, 2);
    l2(i) = 2;
end


x2=zeros(size(Y,2), num_samples2);  % Must be the size of the kernel matrix
for i=1:num_samples2
    x2(:,i) = getCoeffs(x2(:,i), Y2(:,i), Y, ranked_kappa, lambda, eta, 5000, 0);
    predictions(i, 1) = calcZis(x2(:,i), Y2(:,i), Y, ranked_kappa, eta, l2);
end
pred_mask = (predictions'==l2)';
display(['Predicted: ' num2str(100*sum(predictions'==l2)/numel(l2)) '%'])
