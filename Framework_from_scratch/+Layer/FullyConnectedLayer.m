classdef FullyConnectedLayer < Layer.AbstractLayer
    properties % Learnable parameters
        Weight
        Bias
    end
  
    methods
        function layer = FullyConnectedLayer(inLength, outLength)
        % FullyConnectedLayer returns a FullyConnectedLayer
        %
        % Parameters are initialized with random small values
        %
        % Input:
        %   inLength : (Required)
        %       vector length of layer input -- size(X,1) 
        %   outLength : (Required)
        %       vector length of layer output -- size(Z,1)
            layer.Weight = 1e-1 * rand(inLength, outLength);
            layer.Bias   = 1e-1 * rand(outLength, 1);
        end
    
        function Z = forward(layer, X)
            assert(length(size(X)) <= 2, "%s accepts vector input only", mfilename)
            
            Z = layer.Weight' * X + layer.Bias;
        end
    
        function [dLdX, dLdWeight, dLdBias] = backward(layer,X,dLdZ)
            dLdX = layer.Weight * dLdZ;
            
            % sum the gradients over the all samples, division by
            % numSample is already called in the loss function layer
            dLdWeight = X * dLdZ';
            
            dLdBias = sum(dLdZ, 2);
        end
    end
    
    methods
        function params = getLearnableParameters(~)
        % getLearnableParameters tells the optimizer which parameters are
        % trainable
            params = {'Weight', 'Bias'};
        end 
    end
end
