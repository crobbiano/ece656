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

TrainPrice = price(1:floor(end*1/3));
TestPrice = price(floor(end*1/3+1):end);

clear FileName FilePath File data fd

%% Build data matrix
N=3;
M=1;
display(['Using ' num2str(N) 'th order AR model'])
TrainX = zeros(N+M, floor(numel(TrainPrice)/(N+M)));
TrainX = zeros(N+M, numel(TrainPrice));
idx = 0;
for i=1:size(TrainX,2)-(N+M)
%     idx = idx + N+M;
    TrainX(:,i) = TrainPrice(i:i+N+M-1);
end

Rxx = corr(TrainX(1:N,:)');
Rxd =  corr(TrainX(1:N,:)', TrainX(N+1:N+M,:)');
weights_optimal = Rxx\Rxd;

%% Use a Nth order AR model and LMS algorithm to find the coefficients
%  of the AR model.  Compare the results from the LMS with the results
%  using the normal equation
numAttempts = 1;
weights_record = zeros(N, numAttempts);
weights_prev = zeros(N, M);
weights_prev2 = zeros(N, M);
for kk = 1:numAttempts
    costs = Inf(numAttempts, 1);
    costs2 = Inf(numAttempts, 1);
%     weights = -1 + 2*rand(N, M);
%     weights = weights/norm(weights);
    weights = zeros(N, M);
    cost = zeros(2,1);
    cost2 = zeros(2,1);
    cost(1) = 1.2*max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
    cost(2) = 1.1*max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
    cost2(1) = 1.2*immse(TrainX(N+1:N+M,:), weights'*TrainX(1:N,:));
    cost2(2) = 1.1*immse(TrainX(N+1:N+M,:), weights'*TrainX(1:N,:));
    cost_thresh = .000001;
    cost_idx = 2;
    learning_rate = .00000001;
    % while (cost(cost_idx-1) - cost(cost_idx) > .000003)
    while ( cost2(cost_idx) > cost_thresh)
         
        if ((abs(cost2(cost_idx-1) - cost2(cost_idx)) < .0000000001) || abs(cost2(cost_idx) > cost2(cost_idx-1)))
        
            % Reset the weight and cost index to previous iteration and
            % reduc learning rate
            learning_rate = learning_rate / 10;
            weight_prev = weights;
            weights = weights_prev2;
            cost2(cost_idx) = cost_thresh+.001*cost_thresh;
            cost_idx = cost_idx - 1;     
            if learning_rate <= 1e-30
                if abs(cost2(cost_idx) > cost2(cost_idx-1))
                    display('broken')
                else
                    display('not broken')
                end
                break
            end
        end
        cost_idx = cost_idx + 1;
        cost(cost_idx) = max(max((TrainX(N+1:N+M,:) - weights'*TrainX(1:N,:)).^2));
        cost2(cost_idx) = immse(TrainX(N+1:N+M,:), weights'*TrainX(1:N,:));
        if (cost(cost_idx) > cost(cost_idx-1))
%             display('Cost went up. Descent property DNE.  Try smaller learning rate')
        end
        weights_prev2 = weights_prev;
        weights_prev = weights;
        weights = lms_weight_update(TrainX(1:N,:), weights, TrainX(N+1:N+M,:), learning_rate);
        if (mod(cost_idx,1000) == 0)
            display(['Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx)) ' MSE: ' num2str(cost2(cost_idx)) ])
        end
       
    end
    display(['Loop: ' num2str(kk) ' Epoch: ' num2str(cost_idx) ', Cost: ' num2str(cost(cost_idx)) ' MSE: ' num2str(cost2(cost_idx)) ])
    costs(kk) = cost(cost_idx);
    costs2(kk) = cost2(cost_idx);
    weights_record(:, kk) = weights;
end

%% Find best from runs
[best, best_idx] = min(costs2);
best_weights = weights_record(:, best_idx);
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

immse_opt = immse(answers, predicted_optimal);
immse_lms = immse(answers, predicted);

display(['MSE LMS: ' num2str(immse_lms)]);display([ 'MSE Optimal: ' num2str(immse_opt)])
%% Plot Stuff
figure(1); clf;
subplot(2,1,1); 
plot(cost)
title('Learning Rate')
xlabel('Epoch')
ylabel('Cost')

subplot(2,1,2); hold on;
% plot(zscore(answers(:,1)), '-')
% plot(zscore(predicted), '--')
% plot(zscore(predicted_optimal(:,:)))
plot((answers(:,1)), '-')
plot((predicted), '--')
plot((predicted_optimal(:,:)))
title('Price Predictions')
xlabel('Time')
ylabel('Price')
legend('True','LMS Estimate', 'Optimal')


%% WRLS search
%% Use a Nth order AR model and RLS algorithm to find the coefficients
%  of the AR model.  Compare the results from the LMS with the results
%  using the normal equation
numAttemptsRLS = 1;
weights_recordRLS = zeros(N, numAttemptsRLS);
weights_prevRLS = zeros(N, M);
for kk = 1:numAttemptsRLS
    costsRLS = zeros(numAttemptsRLS, 1);
    weightsRLS = -1 + 2*rand(N, M);
%     weights = weights/norm(weights);
%     weights = zeros(N, M);
    costRLS = zeros(2,1);
    costRLS(1) = 1.2*max(max((TrainX(N+1:N+M,:) - weightsRLS'*TrainX(1:N,:)).^2));
    costRLS(2) = 1.1*max(max((TrainX(N+1:N+M,:) - weightsRLS'*TrainX(1:N,:)).^2));
    cost_threshRLS = .000000000001;
    cost_idxRLS = 2;
    learning_rateRLS = .000001;
    
    P = .5*eye(N);
    gamma = 0.75;
    a = 1.5;
    % while (cost(cost_idx-1) - cost(cost_idx) > .000003)
    while ( costRLS(cost_idxRLS) > cost_threshRLS)
        if ((abs(costRLS(cost_idxRLS-1) - costRLS(cost_idxRLS)) < .0000000000000001) || costRLS(cost_idxRLS) > costRLS(cost_idxRLS-1))
            % Reset the weight and cost index to previous iteration and
            % reduc learning rate
            break
        end
        cost_idxRLS = cost_idxRLS + 1;
        costRLS(cost_idxRLS) = max(max((TrainX(N+1:N+M,:) - weightsRLS'*TrainX(1:N,:)).^2));
        if (costRLS(cost_idxRLS) > costRLS(cost_idxRLS-1))
%             display('Cost went up. Descent property DNE.  Try smaller learning rate')
        end
        for n=1:size(TrainX(1:N,:),2)
            if TrainX(1:N,n)'*weightsRLS <= 0
                net = 0;
            elseif TrainX(1:N,n)'*weightsRLS >= a
                net = 1;
            else
                net = TrainX(1:N,n)'*weightsRLS;
            
            gain = P*TrainX(1:N,n)/(gamma + TrainX(1:N,n)'*P*TrainX(1:N,n));
            P = (1/gamma)*(eye(N) - gain*TrainX(1:N,n)')*P;
            weightsRLS = weightsRLS + gain*(TrainX(N+1:N+M,n) - (1/a)*net);
            end
        end
%         [weights, P] = wrls_weight_update(TrainX(1:N,:), weights, .99, P, TrainX(N+1:N+M,:)', 50);
%         if (mod(cost_idxRLS,1000) == 0)
            display(['Epoch: ' num2str(cost_idxRLS) ', Cost: ' num2str(costRLS(cost_idxRLS)) ])
%         end
    end
    display(['Loop: ' num2str(kk) ' Epoch: ' num2str(cost_idxRLS) ', Cost: ' num2str(costRLS(cost_idxRLS)) ])
    costsRLS(kk) = costRLS(cost_idxRLS);
    weights_recordRLS(:, kk) = weightsRLS;
end
%% Plot Stuff
% Find best from runs
[bestRLS, best_idxRLS] = min(costsRLS);
best_weightsRLS = weights_recordRLS(:, best_idxRLS);
%  Use the best weights to do some predictions
pred_idxRLS = 1;
% endLen = inputDataLen-(N-1);
predLenRLS = 400;
endLenRLS = predLenRLS;
predictedRLS = zeros(predLenRLS,M);
predicted_optimalRLS = zeros(predLenRLS,M);
answersRLS = zeros(predLenRLS,M);
for i=1:endLenRLS
    % Form the inputs
    inputs = TestPrice(i:i+(N-1));
    answersRLS(pred_idxRLS,:) = TestPrice(i+N:i+N+(M-1));
    predictedRLS(pred_idxRLS,:) = weights_recordRLS(:,best_idxRLS)'*inputs;
    predicted_optimalRLS(pred_idxRLS,:) = weights_optimal'*inputs;
    pred_idxRLS = pred_idxRLS + 1;
end

figure(2); clf;
% subplot(2,2,1); 
plot(cost)
title('LMS Learning Rate')
xlabel('Epoch')
ylabel('Cost')

figure(3); clf
% subplot(2,2,2); 
plot(costRLS)
title('RLS Learning Rate')
xlabel('Epoch')
ylabel('Cost')

figure(4); clf
% subplot(2,2,[3, 4]); 
hold on;
% plot(zscore(answers(:,1)), '-')
% plot(zscore(predicted), '--')
% plot(zscore(predicted_optimal(:,:)))
plot((answersRLS(:,1)), '-')
plot((predictedRLS), '--')
plot((predicted), '--.')
plot((predicted_optimalRLS(:,:)))
title('Price Predictions')
xlabel('Time')
ylabel('Price')
legend('True','RLS Estimate','LMS Estimate', 'Optimal')
xlim([190 210])

immse_opt = immse(answersRLS, predicted_optimalRLS);
immse_rls = immse(answersRLS, predictedRLS);

display(['MSE LMS: ' num2str(immse_lms)]);display(['MSE RLS: ' num2str(immse_rls)]);display([ 'MSE Optimal: ' num2str(immse_opt)])

%% Quantify the error
error_opt = answers - predicted_optimalRLS;
error_lms = answers - predicted;
error_rls = answers - predictedRLS;
figure(6); clf;
h0 = histfit(error_opt)
figure(7); clf;
h1 = histfit(error_lms)
figure(8); clf;
h2 = histfit(error_rls)

%% Look at xcorr
[acor,lag] = xcorr(price);
[~,I] = max(abs(acor));
timeDiff = lag(I)         % sensor 2 leads sensor 1 by 350 samples
figure(19); clf
plot(lag,acor);
grid
title('Autocorrelation function for data')