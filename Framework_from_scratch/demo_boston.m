import Layer.* Utils.* Dataloader.*

% 1. load data
[X,Y] = readData('boston');
X = normalize(X,2);

% 2. set training options
trainingOptions = struct(...
    'Optimizer', 'sgdm',...
    'InitialLearnRate', 1e-1, ...
    'Momentum', 0.9,...
    'LearnRateDropPeriod', 3000, ...
    'MaxEpoch',5000, ...
    'BatchSize', 128, ...
    'LearnRateDropFactor', 0.1, ...
    'GradientThreshold', 1,...
    'UseGPU', false, ...
    'Shuffle', true, ...
    'Verbose', true, ...
    'VerboseFrequency', 100);

% 3. initialize network
InputLength  = size(X,1);
OutputLength = size(Y,1);
Net = {...
    FullyConnectedLayer(InputLength,50),...
    ReluLayer(),...
    FullyConnectedLayer(50,OutputLength),...
    MSERegressionLayer() };

% 4. train network
Net = trainNetwork(Net, X, Y, trainingOptions);

% 5. test trained network
Y_hat = predict(Net, X);

hfig = figure;
NumSample = size(Y,2);
plot(1:NumSample,Y,'ro',...
   1:NumSample,Y_hat,'b+');
xlabel('x');ylabel('y');title('prediction');