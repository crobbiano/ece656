%Convolutional neural network for handwriten digits recognition: training
%and simulation.
%(c)Mikhail Sirotenko, 2009.
%This program implements the convolutional neural network for MNIST handwriten
%digits recognition, created by Yann LeCun. CNN class allows to make your
%own convolutional neural net, defining arbitrary structure and parameters.
%It is assumed that MNIST database is located in './MNIST' directory.
%References:
%1. Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G.
%and Muller K. (Eds), Neural Networks: Tricks of the trade, Springer, 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%2. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning
%Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%3. Patrice Y. Simard, Dave Steinkraus, John C. Platt: Best Practices for
%Convolutional Neural Networks Applied to Visual Document Analysis
%URL:http://research.microsoft.com/apps/pubs/?id=68920
%4. Thanks to Mike O'Neill for his great article, wich is summarize and
%generalize all the information in 1-3 for better understandig for
%programming:
%URL: http://www.codeproject.com/KB/library/NeuralNetRecognition.aspx
%5. Also thanks to Jake Bouvrie for his "Notes on Convolutional Neural
%Networks", particulary for the idea to debug the neural network using
%finite differences
%URL: http://web.mit.edu/jvb/www/cv.html

clear all;
clc;
close all
%%
imds = imageDatastore('D:\Documents\ece656\compass2\files', 'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%% 3 layer
layers3 = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

[net3, tr3] = trainNetwork(imdsTrain,layers3,options);

YPred = classify(net3,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
figure(17); plotconfusion(YValidation,YPred);
testt = grp2idx(YValidation);
testt = testt - 1;
testt10 = num2bin10(testt);
pred = grp2idx(YPred);
pred = pred - 1;
pred = num2bin10(pred);
figure(5), plotroc(testt10,pred)
%% 2 lyer
layers2 = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

[net2, tr2] = trainNetwork(imdsTrain,layers2,options);

YPred = classify(net2,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
figure(18); plotconfusion(YValidation,YPred);testt = grp2idx(YValidation);
testt = grp2idx(YValidation);
testt = testt - 1;
testt10 = num2bin10(testt);
pred = grp2idx(YPred);
pred = pred - 1;
pred = num2bin10(pred);
figure(6), plotroc(testt10,pred)