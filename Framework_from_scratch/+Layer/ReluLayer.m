classdef ReluLayer < Layer.AbstractLayer
    
    methods
        function Z = forward(~,X)
            Z = max(0, X);
        end
        
        function dLdX = backward(~,X,dLdZ)
            dLdX = zeros(size(dLdZ), 'like', dLdZ);
            dLdX(X>0) = dLdZ(X>0);
        end
    end
end
