%% Computer Assignment 1 - ECE656
% Supervised learning for linear prediction of stock markets
% Using the futures data from our other trading project

clear all
clc

%% Load the data
FileName = 'ES_all-15_16_17.Last.txt';
FilePath = '..\data';
File = [FilePath '\' FileName];

% File Format: yyyyMMdd HHmmss;open price;high price;low price;close price;volume
fd = fopen(File,'rt');
data = textscan(fd, '%d%d%f%f%f%f%d', 'Whitespace', '; \t');
fclose(fd);

% Generating useable data array
price  = cell2mat(data(:, 6));      % close Price
price = zscore(price);
inputDataLen = length(price);       % length of the input data, used in 'start' var below

TrainPrice = price(1:floor(end/3));
TestPrice = price(floor(end/3+1):end);

clear FileName FilePath File data fd

%% Build data matrix
N=5;
M=1;
display(['Using ' num2str(N) 'th order AR model'])
TrainX = zeros(N+M, floor(numel(TrainPrice)/(N+M)));
idx = 0;
for i=1:size(TrainX,2)-1
    idx = idx + N+M;
    TrainX(:,i) = TrainPrice(idx:idx+N+M-1);
end

Rxx = corr(TrainX(1:N,:)');
Rxd =  corr(TrainX(1:N,:)', TrainX(N+1:N+M,:)');
weights_optimal = Rxx\Rxd;

%% Use a Nth order AR model and LMS algorithm to find the coefficients
%  of the AR model.  Compare the results from the LMS with the results
%  using the normal equation
numAttempts = 200;
weights_record = zeros(N, numAttempts);
costs = zeros(numAttempts, 1);
for kk = 1:numAttempts
    weights = -1 + 2*rand(N, M);
%     weights = weights/norm(weights);
%     weights = zeros(N, M);
    weights_record(:, kk) = weights;
    cost = zeros(2,1);
    cost(1) = 1.2*max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
    cost(2) = 1.1*max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
    cost_thresh = .01;
    cost_idx = 2;
    learning_rate = .000001;
    % while (cost(cost_idx-1) - cost(cost_idx) > .000003)
    while ( cost(cost_idx) > cost_thresh)
        if ((abs(cost(cost_idx-1) - cost(cost_idx)) < .000001) || cost(cost_idx) > cost(cost_idx-1))
            learning_rate = learning_rate / 10;
            if learning_rate <= 1e-30
                break
            end
        end
        cost_idx = cost_idx + 1;
        cost(cost_idx) = max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
        if (cost(cost_idx) > cost(cost_idx-1))
%             display('Cost went up. Descent property DNE.  Try smaller learning rate')
        end
        weights = lms_weight_update(TrainX(1:N,:), weights, TrainX(N+1:N+M,:), learning_rate);
        if (mod(cost_idx,1000) == 0)
            display(['Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx))])
        end
    end
    display(['Loop: ' num2str(kk) ' Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx))])
    costs(kk) = cost(cost_idx);
end

%% Find best from runs
[best, best_idx] = min(costs)
best_weights = weights_record(:, best_idx)
% best_weights = best_weights/norm(best_weights)

%% Use the best weights to do some predictions
pred_idx = 1;
% endLen = inputDataLen-(N-1);
predLen = 400;
endLen = predLen;
predicted = zeros(predLen,M);
predicted_optimal = zeros(predLen,M);
answers = zeros(predLen,M);
for i=1:endLen
    % Form the inputs
    inputs = TestPrice(i:i+(N-1));
    answers(pred_idx,:) = TestPrice(i+N:i+N+(M-1));
    predicted(pred_idx,:) = weights_record(:,best_idx)'*inputs;
    predicted_optimal(pred_idx,:) = weights_optimal'*inputs;
    pred_idx = pred_idx + 1;
end


%% Plot Stuff
figure(1); clf;
subplot(2,1,1); 
plot(cost)
title('Learning Rate')
xlabel('Epoch')
ylabel('Cost')

subplot(2,1,2); hold on;
plot(zscore(answers(:,1)), '-')
plot(zscore(predicted), '--')
plot(zscore(predicted_optimal(:,:)))
title('Price Predictions')
xlabel('Time')
ylabel('Price')
legend('True','LMS Estimate', 'Optimal')
