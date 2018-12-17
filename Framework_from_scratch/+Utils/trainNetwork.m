function net = trainNetwork(net, X, Y, option)
% trainNetwork trains a deep neural network
%
% Input:
%   net    : (Required) cell
%   X      : (Required) 2-D Array
%       input of training data
%   Y      : (Required) 2-D Array
%       groudtruth/label of training data
%   option : (Required) Struct
%       options for training network
    import Utils.* Utils.Training.*
    import Optimizer.initializeOptimizer
    import Dataloader.onehot2gray
    
    % simple validation on inputs
    assert(iscell(net), 'Network should be cell of layers');
    assert(ismatrix(X), "Currently %s accepts 2-D input only, instead X is %d-D", mfilename, length(size(X)));
    assert(ismatrix(Y), "Currently %s accepts 2-D input only, instead Y is %d-D", mfilename, length(size(Y)));
    
    Optimizer           = option.Optimizer;
    InitialLearnRate    = option.InitialLearnRate;
    Momentum            = option.Momentum;
    GradientThreshold   = option.GradientThreshold;
    LearnRateDropPeriod = option.LearnRateDropPeriod;
    LearnRateDropFactor = option.LearnRateDropFactor;
    
    BatchSize           = option.BatchSize;
    MaxEpoch            = option.MaxEpoch;
    Verbose             = option.Verbose;
    VerboseFrequency    = option.VerboseFrequency;
    UseGPU              = option.UseGPU;
    Shuffle             = option.Shuffle;
    
    optimizerOptions = struct(...
        'LearnRate', InitialLearnRate,...
        'Momentum',  Momentum,...
        'GradientThreshold', GradientThreshold);
    optimizer = initializeOptimizer(Optimizer, optimizerOptions);
     
    LearnRates = LearnRateScheduler(InitialLearnRate, MaxEpoch, LearnRateDropPeriod, LearnRateDropFactor);
    
    if Shuffle 
        % This is slow since we swap the actual data instead of the indices
        % However, this is acceptable since we only do it once
        I = randperm(size(X,2));
        X = X(:,I);
        Y = Y(:,I);
    end
    
    if UseGPU % TODO: GPU is slower than CPU in current version
        % move all data to GPU
        warning("currently using GPU is slower than using CPU");
        X = gpuArray(X);
        Y = gpuArray(Y);
        net = gpuNetwork(net);
    end
    
    if Verbose
       numSample = min(BatchSize, size(X,2));
       preview_X = X(:, 1:numSample); % used for Verbose
       preview_Y = Y(:, 1:numSample);
       
       isOnehot = size(preview_Y,1) ~= 1;
       
       hfig = figure;

       subplot(1,2,1);
       hfig_loss = animatedline();
       xlabel('iter');ylabel('error');
       title('Gradient Descent Progress')
    end
    
    NumSample = size(X,2);
    iter = 1;
    for epoch = 1:MaxEpoch % epoch iteration
        optimizer.LearnRate = LearnRates(epoch);
        
        indices = generateBatchIndices(NumSample, BatchSize);
        numBatches = length(indices);
        for i = 1: numBatches % mini-batch iteration
            cur_X = X(:, indices{i});
            cur_Y = Y(:, indices{i});
            
            % forward + backward + update
            [Z, loss] = forwardLoss(net, cur_X, cur_Y);
            [~, dLdParams] = backwardLoss(net, Z, cur_Y);
            update(optimizer, net, dLdParams);
            
            % show training progress
            if Verbose && mod(iter, VerboseFrequency) == 0
                addpoints(hfig_loss, iter, loss);

                figure(hfig);subplot(1,2,2);
                if isOnehot
                    plot(1:size(preview_Y,2),onehot2gray(preview_Y),'ro',...
                       1:size(preview_Y,2),onehot2gray(predict(net,preview_X)),'b+');
                else
                    plot(1:size(preview_Y,2),preview_Y,'ro',...
                       1:size(preview_Y,2),predict(net,preview_X),'b+');
                end
                xlabel('x');ylabel('y');title('prediction');

                drawnow;

                fprintf(sprintf("Epoch: %d Iter: %d Loss: %f Learning Rate: %f\n", ...
                    epoch, iter, loss, optimizer.LearnRate));
            end
            
            iter = iter + 1;
        end
    end
    
    if UseGPU
        net = gatherNetwork(net);
    end
end

function indices = generateBatchIndices(n, BatchSize)
% generateBatchIndices generate a cell of indices for selecting mini-batch
% of training data
%
% Output:
%   indices : cell of array
%       X(indices{i}) is the i-th minibatch of training data X
    I = randperm(n);
    numBatch = ceil(n/BatchSize);
    
    indices = cell(numBatch,1);
    for i = 1:numBatch
        l = (i-1) * BatchSize + 1;
        r = min(i * BatchSize, n);
        indices{i} = I(l:r);
    end
end