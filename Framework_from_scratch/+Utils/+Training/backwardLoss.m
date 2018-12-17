function [dLdX, dLdParams] = backwardLoss(net, Z, Y)
% backwardLoss backward propogates the gradients of the network
%
% Input:
%   net : (Required) cell
%   Z   : (Required) cell
%       Z{i} is the input of the i-th layer
%       Z{i+1} is the output of the i-th layer
%   Y   : (Required) 2-D array
%       network training label
% Output:
%   dLdX      : 2-D array
%       dLdX{i} is the derivative of loss with regard to input X, i.e., Z{1}
%   dLdParams : cell of cell
%       * dLdParams{i} is a cell that contains the derivative of loss with 
%       regard to learnable parameters in i-th layer;
%       * dLdParams{i}{j} is the derivative of loss with regard to j-th
%       learnable parameter in i-th layer;
%       * if i-th layer has no learnable parameter (e.g., ReluLayer), then 
%       dLdParams{i} is an empty cell.   
    assert(iscell(net), 'Network should be cell of layers');
    assert(iscell(Z));
    assert(ismatrix(Y)); % 2-D array
    
    dLdParams = cell(length(net),1);
    
    dLdZ = net{end}.backwardLoss(Z{end}, Y);
    for i = length(net)-1:-1:1
        [dLdZ, dLdParams{i}] = backward_(net{i}, Z{i}, dLdZ);
    end
    
    dLdX = dLdZ;
    
    assert(ismatrix(dLdX)); % 2-D array
    cellfun(@(x) assert(iscell(x) | isempty(x)), dLdParams); % cell of cell
end

function [dLdX, dLdParam] = backward_(layer, X, dLdZ)
    dLdParam = {};
    try
        % layer with learnable parameters
        [dLdX, varargout{1:nargout}] = layer.backward(X,dLdZ);
        dLdParam = varargout;
    catch err
        if strcmp(err.identifier, "MATLAB:TooManyOutputs")
            % layer without learnable parameters
            dLdX = layer.backward(X,dLdZ); % TODO: backward is executed two times here, which is uncessary.
        else
            rethrow(err);
        end
    end
end