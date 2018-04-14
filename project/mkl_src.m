%% MLK - SRC
clear all
close all
clc
%% Load some data
% Load images and labels

% Build data sample matrix Y and label vector l
%% Make kernel functions
% Choose the kernel functions and make vector of them

%% Compute the M kernel matrices
% Find K_m(Y, Y) for all M kernel functions

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
N = num_samples;

%% Find initial sparse coefficients matrix X
% For each ith training sample, 0 out the ith row
% of Y and call it Y_tilde, then solve for x_i in y_i=(Y_tilde)x_i
% by solving (18) in the paper. 
% x_i = argmin_x (k(y_i, y_i)+ x^TK(Y_tilde, Y_tilde)x - 2K(y_i, Y_tilde)x - lambda||x_i||^1)




%% Iterate until quitting conditions are satisfied
while(t <= T && err>= err_thresh)
    % Compute K_m(Y_tilde, Y_tilde)
    for i=1:N
        Y_tilde = Y;
        % Zero ith column of Y
        Y_tilde(:,i) = 0;
    end
    
    % Compute the sparse code x_i as done before - MAKE FNC FOR THIS
    
    % Compute the predicted label g_i using x_i
end