classdef LeakyReluLayer < Layer.AbstractLayer
    
    properties
        Alpha
    end
  
    
    methods
        function layer = LeakyReluLayer(Alpha)
            layer.Alpha = Alpha;
        end

        
        function Z = forward(layer,X)
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
        
        
        function dLdX = backward(layer,X,dLdZ)
            dLdX = layer.Alpha .* dLdZ;
            dLdX(X>0) = dLdZ(X>0);
        end
    end
end