function [Z, loss] = forwardLoss(net, X, Y)
% forwardLoss forward propogates the outputs of the network
%   
% Input:
%   net : (Required) cell
%   X   : (Required) 2-D array
%       network input
%   Y   : (Required) 2-D array
%       network training label
% Output:
%   Z    : cell
%       Z{i} is the input of the i-th layer
%       Z{i+1} is the output of the i-th layer
%   loss : scalar
    assert(iscell(net), 'Network should be cell of layers');
    
    Z = cell(length(net),1);
    Z{1} = X;
    for i = 1:length(net)-1
        Z{i+1} = net{i}.forward(Z{i});
    end
    
    loss = net{end}.forwardLoss(Z{end},Y);
end