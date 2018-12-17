classdef (Abstract) AbstractLossLayer < handle
    properties (Access = private)
        cache
    end
    
    
    methods (Abstract)
        loss = forwardLoss(layer, Y, T);
        dLdZ = backwardLoss(layer, Y, T);
    end
end



