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
inputDataLen = length(price);       % length of the input data, used in 'start' var below

TrainPrice = price(1:floor(end/3));
TestPrice = price(floor(end/3+1):end);

clear FileName FilePath File data fd

%% Build data matrix
N=10;
display(['Using ' num2str(N) 'th order AR model'])
TrainX = zeros(N+1, floor(numel(TrainPrice)/(N+1)));
idx = 0;
for i=1:size(TrainX,2)-1
    idx = idx + N+1;
    TrainX(:,i) = TrainPrice(idx:idx+N);
end

Rxx = corr(TrainX(1:N,:)');
Rxd =  corr(TrainX(1:N,:)', TrainX(N+1,:)');
weights_optimal = inv(Rxx)*Rxd;

%% Use a Nth order AR model and LMS algorithm to find the coefficients
%  of the AR model.  Compare the results from the LMS with the results
%  using the normal equation

weights = zeros(N, 1);
cost(1) = 100*mean((TrainX(N+1,:) - weights'*TrainX(1:N,:)).^2);
cost(2) = 50*mean((TrainX(N+1,:) - weights'*TrainX(1:N,:)).^2);
cost_thresh = .6;
cost_idx = 2;
learning_rate = .0000000000001;
while (cost(cost_idx-1) - cost(cost_idx) > .0001)    
    cost_idx = cost_idx + 1;
    cost(cost_idx) = mean((TrainX(N+1,:) - weights'*TrainX(1:N,:)).^2);
    if (cost(cost_idx) > cost(cost_idx-1))
        error('Cost went up. Descent property DNE.  Try smaller learning rate')
    end
    weights = lms_weight_update(TrainX(1:N,:), weights, TrainX(N+1,:)', learning_rate);
    if (mod(cost_idx,10) == 0)
        display(['Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx))])
    end
end
display(['Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx))])

%% Use the learned weights to do some predictions
pred_idx = 1;
% endLen = inputDataLen-(N-1);
predLen = 30000;
endLen = predLen;
predicted = zeros(predLen,1);
predicted_optimal = zeros(predLen,1);
for i=1:endLen
    % Form the inputs
    inputs = TestPrice(i:i+(N-1));
    predicted(pred_idx) = weights'*inputs;
    predicted_optimal(pred_idx) = weights_optimal'*inputs;
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
plot(TestPrice((N):endLen+(N)), '-r')  % Need this shift to be consistent inputs from above
plot(predicted, '--g')
plot(predicted_optimal, '-.b')
title('Price Predictions')
xlabel('Time')
ylabel('Price')
legend('True','LMS Estimate', 'Optimal')
