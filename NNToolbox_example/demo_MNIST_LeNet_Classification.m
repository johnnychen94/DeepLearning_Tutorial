% This demo trains a LeNet to classify MNIST images
%
% How to:
% 1. Download MNIST Database, unzip it, and then copy the four files to ./Datasets/MNIST;
% 2. run this file in MATLAB
%
% see https://johnnychen94.github.io/tutorial/2017/12/23/MNIST_LeNet_Matlab.html
% for more details
%
% Copyright (C) 2017 Johnny Chen  
% Email: johnnychen94@hotmail.com
% 

% Global settings
s = rng('shuffle'); % set random seed
use_pretrained_net = true; % true to use existed pretrained net
data_path = fullfile('Datasets','MNIST'); % MNIST database file folder
net_cache_path = fullfile('Trained_Networks','MNIST_LeNet.mat'); % 


% load MNIST Dataset
[train_imgs,train_labels] = Dataloaders.MNIST_reader(data_path,'train',true);
[test_imgs,test_labels] = Dataloaders.MNIST_reader(data_path,'train',false);
labels = unique(test_labels);

% Preview
p = randperm(length(train_imgs),20);
figure('Name','Preview of MNIST')
for i = 1:20
  subplot(4,5,i),imshow(train_imgs(:,:,1,p(i)),[]),title(string(train_labels(p(i))))
end
pause(0.1);

if use_pretrained_net && isfile(net_cache_path)
  % use pretrained network
  trained_net = load(net_cache_path,'trained_net');
  trained_net = trained_net.trained_net;
else
  % train from scratch

  % define network architecture
  C1 = [...
        convolution2dLayer([5,5],6,'Name','C1');
        reluLayer('Name','ReLu1');];
  S2 = maxPooling2dLayer(2,'Stride',2,'Name','S2');
  C3 = [...
        convolution2dLayer([5,5],16,'Name','C3');
        reluLayer('Name','ReLu3');];
  S4 = maxPooling2dLayer(2,'Stride',2,'Name','S4');
  C5 = [...
        fullyConnectedLayer(120,'Name','C5');
        reluLayer('Name','ReLu5');];
  F6 = [...
        fullyConnectedLayer(84,'Name','F6');
        reluLayer('Name','ReLu6');];
  Output = fullyConnectedLayer(10,'Name','Output');

  layers = [...
        imageInputLayer([28,28,1],'Name','Input');
        C1;S2;C3;S4;C5;F6;
        Output;
        softmaxLayer('Name','Softmax');
        classificationLayer('Name','Classification');
        ];

  % show network
  figure('Name','LeNet Graph'),
  lgraph = layerGraph(layers);
  plot(lgraph),title('LeNet Graph');
  pause(0.1);

  % set training options
  training_options = trainingOptions('sgdm',...
                                            'Momentum',0.9,...
                                            'L2Regularization',1e-4,...
                                            'MaxEpoch',5,...
                                            'Shuffle','once',...
                                            'LearnRateSchedule','piecewise',...
                                            'LearnRateDropFactor',0.1,...
                                            'LearnRateDropPeriod',1,...
                                            'ValidationData',{test_imgs,test_labels},...
                                            'ValidationPatience',5,...
                                            'Plots','training-progress');

  % training network
  % takes about 5 mins for CPU, 30 seconds for GPU
  trained_net = trainNetwork(train_imgs,train_labels,layers,training_options);
  save(net_cache_path,'trained_net');
end


% Test network
p = randperm(length(test_imgs),20);
figure('Name','Test Trained Network')
for i = 1:20
  cur_img = test_imgs(:,:,1,p(i));
  [~, cur_predict_index] = max(predict(trained_net,cur_img));
  subplot(4,5,i),
  imshow(cur_img,[]),
  title(string(labels(cur_predict_index)));
end
pause(0.1);

% Count test results
[counts, corrects, test_duration, wrong_imgs] = snippets.test_network(trained_net,test_imgs,test_labels);

% show wrong predicted images
figure('Name','Wrong Predicted Images'),
for i = 1:min(20,length(wrong_imgs))
  cur_img = wrong_imgs{i};
  [~, cur_predict_index] = max(predict(trained_net,cur_img));
  subplot(4,5,i),
  imshow(cur_img,[]),
  title(string(labels(cur_predict_index)));
end
pause(0.1);


% show test results
snippets.show_test_results(counts,corrects,labels,test_duration);
