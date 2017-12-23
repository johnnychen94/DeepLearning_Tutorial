% s = rng('shuffle');
s = rng('shuffle');

% load MNIST Dataset
data_path = fullfile('Datasets','MNIST');
[train_imgs,train_labels] = Dataloaders.MNIST_reader(data_path,'train',true);
[test_imgs,test_labels] = Dataloaders.MNIST_reader(data_path,'train',false);

% Preview
p = randperm(length(train_imgs),20);
figure('Name','Preview of MNIST')
for i = 1:20
  subplot(4,5,i),imshow(train_imgs(:,:,1,p(i)),[]),title(string(train_labels(p(i))))
end
pause(1);


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
lgraph = layerGraph(layers);
plot(lgraph),title('LeNet Graph');

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
                                            'ValidationPatience',Inf,...
                                            'Plots','training-progress');

% training network                                          
trained_net = trainNetwork(train_imgs,train_labels,layers,training_options);



