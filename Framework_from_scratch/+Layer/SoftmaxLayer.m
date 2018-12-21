classdef SoftmaxLayer < Layer.AbstractLayer
    
    methods
        
        function Z = forward(~, X)
            X = X - max(X,[],1);
            expX = exp(X);
            Z = expX./sum(expX);
        end
        
        function [dX, dW] = backward(layer, X, dZ)
            Z = layer.forward(X);
            Z = Layer.Utils.boundAwayFromZero(Z);
            
            dotProduct = sum(Z.*dZ);
            dX = dZ - dotProduct;
            dX = dX.*Z;
            dW = [];
        end
    end
end


