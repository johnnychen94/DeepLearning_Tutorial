
% load training and validation data
[Xtrain,Ytrain,Xvalid,Yvalid] = Dataloader.load_house_price(0,5);
% plot(Xtrain,Ytrain,'x');
% axis([-10,200,-10,400]);

% initialize network
net = init_network();

% set hyperparameter
train_opts.Learningrate = 1e-4;
train_opts.MaxEpochs = 1000;
train_opts.Plots = true;
train_opts.Verbose = true;
train_opts.Momentum = 0.9;

% train network
net = train_network(net,Xtrain,Ytrain,train_opts);

% test network




function net = init_network()
  net = {Layer.FullyConnectedLayer(),...
    Layer.ReluLayer(0.1),...
    Layer.FullyConnectedLayer(),...
    Layer.ReluLayer(0.1),...
    Layer.FullyConnectedLayer(),...
    Layer.ReluLayer(0.1),...
    Layer.MSERegressionLayer()};
end

function net = train_network(net,Xtrain,Ytrain,trainingOptions)
  lr = trainingOptions.Learningrate;
  verbose = trainingOptions.Verbose;
  plots = trainingOptions.Plots;
  momentum = trainingOptions.Momentum;
  
  for i = 1:trainingOptions.MaxEpochs
    cur_X = Xtrain; % use whole data as a minibatch
    cur_Y = Ytrain;
    [cur_Y_est,cur_loss] = forward(net,cur_X,cur_Y);
    
    net = update_params(net,cur_Y_est,cur_Y,lr,momentum);
    
    if verbose
      fprintf('Epoch: %d\t Loss:%.3f\n',i,cur_loss);
    end
    if plots
      plot(cur_X,cur_Y_est{end});
      hold on;
      plot(cur_X,cur_Y,'x');
      hold off;
    end
    pause(0.05);
  end
  
  function [Y_est,loss] = forward(net,X,Y_truth)
    
    Y_est = cell(length(net),1);
    Y_est{1} = X;
    for idx = 1:length(net)-1
      Y_est{idx+1} = net{idx}.forward(Y_est{idx});
    end
    loss = net{end}.forwardLoss(Y_est{idx+1},Y_truth);
  end

  function net = update_params(net,Y_est,Y_truth,lr,momentum)
    dLdphi = net{end}.backwardLoss(Y_est{end},Y_truth);
    
    dLdf = net{end-1}.backward(Y_est{end-1},dLdphi);
    [dLdphi,dLdWeight,dLdBias] = net{end-2}.backward(Y_est{end-2},dLdf);
    net{end-2}.update_params(lr,momentum,dLdWeight,dLdBias);
    
    dLdf = net{end-3}.backward(Y_est{end-3},dLdphi);
    [dLdphi,dLdWeight,dLdBias] = net{end-4}.backward(Y_est{end-4},dLdf);
    net{end-4}.update_params(lr,momentum,dLdWeight,dLdBias);
    
    dLdf = net{end-5}.backward(Y_est{end-5},dLdphi);
    [dLdphi,dLdWeight,dLdBias] = net{end-6}.backward(Y_est{end-6},dLdf);
    net{end-6}.update_params(lr,momentum,dLdWeight,dLdBias);
  end
end

