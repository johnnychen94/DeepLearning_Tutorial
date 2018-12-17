classdef SGDM < Optimizer.AbstractOptimizer
% SGDM is stocastic gradient descent optimizer with momentum
    properties
        LearnRate
        Momentum
        GradientThreshold
    end
    
    methods
        function optimizer = SGDM(LearnRate, Momentum, GradientThreshold)
        % SGDM returns a SGDM (stochastic gradient descent with momentum) optimizer
        %
        % Input:
        %   LearnRate : (Required)
        %   Momentum  : (Required)
        %   GradientThreshold : (Required)
        %       threshold for gradient clipping
            optimizer.LearnRate = LearnRate;
            optimizer.Momentum = Momentum;
            optimizer.GradientThreshold = GradientThreshold;
        end
        
        function update(optimizer, layer, dLdParams)
            params = layer.getLearnableParameters(); % dparams{i} is the gradient of params{i}
            if isempty(params) % no parameter to update
                return
            end
            
            velocity = layer.cache();
            if isempty(velocity) %  initialization for the first time
                velocity = optimizer.initializeCache(dLdParams);
            end
            
            for i = 1:length(params) % update each parameter of given layer
                p = params{i};
                velocity{i} = (1-optimizer.Momentum) * dLdParams{i} + optimizer.Momentum * velocity{i};
                layer.(p) = layer.(p) - optimizer.LearnRate * gradientClip(velocity{i}, optimizer.GradientThreshold);
            end
            
            layer.cache = velocity;
        end
    end
end

function grad = gradientClip(grad, thres)
% gradientClip helps avoid NAN gradient
    grad = min(max(grad, -thres),thres);
end

