classdef (Abstract) AbstractLayer < handle
    properties
        cache
    end
    
    
    methods (Abstract) % API for every layer
        Z = forward(layer, X);
        [dLdX, varargout] = backward(layer, X, dLdZ);
    end
    
    
    methods
        function Z = predict(layer, X)
            % some particular layer has different forward behavior in 
            % training and testing stage
            Z = forward(layer, X);
        end
    end
    
    
    methods % API for optimizers
        function params = getLearnableParameters(~)
        % getLearnableParameters tells the optimizer which parameters are
        % trainable
        % 
        % Note: override this function to identify the learnable parameter
            params = {};
        end
    end
end



