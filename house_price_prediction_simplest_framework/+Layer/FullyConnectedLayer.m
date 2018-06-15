classdef FullyConnectedLayer < handle
  properties
    Weight
    Bias
  end
  properties(Access=private)
    vdWeight % velocity of dWeight
    vdBias % velocity of dBias
  end
  methods
    function layer = FullyConnectedLayer()
      layer.Weight = 2+randn();
      layer.Bias = -50+randn();
      layer.vdWeight = 0;
      layer.vdBias = 0;
    end
    
    function Z = forward(layer,X)
      Z = layer.Weight * X + layer.Bias;
    end
    
    function [dLdX,dLdWeight,dLdBias] = backward(layer,X,dLdZ)
      dLdX = dLdZ * layer.Weight;
      dLdWeight = sum(dLdZ .* X);
      dLdBias = sum(dLdZ);
    end
    
    function update_params(layer,lr,momentum,dLdWeight,dLdBias)
      % sgdm
      layer.vdWeight = (1-momentum)*dLdWeight + momentum * layer.vdWeight;
      layer.Weight = layer.Weight - lr * layer.vdWeight;
      
      layer.vdBias = (1-momentum)*dLdBias + momentum * layer.vdBias;
      layer.Bias = layer.Bias - lr * layer.vdBias;
    end
  end
end
