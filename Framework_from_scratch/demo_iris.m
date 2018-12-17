import Layer.* Utils.* Dataloader.*

% 1. load data
[X,Y] = readData('iris');
% Y = onehot2gray(Y);
X = normalize(X,2);

% 2. set training options
traning_option = trainingOptions('Optimizer', 'sgdm',...
    'InitialLearnRate', 1e-2, ...
    'LearnRateDropPeriod', 4000, ...
    'MaxEpoch',5000, ...
    'BatchSize', 128, ...
    'Verbose', true, ...
    'VerboseFrequency', 20,...
    'GradientThreshold', 1);

% 3. initialize network
InputLength  = size(X,1);
OutputLength = size(Y,1);
Net = {FullyConnectedLayer(InputLength,50),...
    LeakyReluLayer(0.2),...
    FullyConnectedLayer(50,OutputLength),...
    MSERegressionLayer()};

% 4. train network
Net = trainNetwork(Net, X, Y, traning_option);

% 5. test trained network
Y_hat = predict(Net, X);

hfig = figure;
NumSample = size(Y,2);

isOnehot = size(Y,1) ~= 1;
if isOnehot
    plot(1:NumSample,onehot2gray(Y),'ro',...
        1:NumSample,onehot2gray(Y_hat),'b+');
else
    plot(1:NumSample,Y,'ro',...
        1:NumSample,Y_hat,'b+');
end
xlabel('x');ylabel('y');title('prediction');
