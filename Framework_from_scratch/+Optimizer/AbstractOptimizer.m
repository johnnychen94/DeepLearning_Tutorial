classdef (Abstract) AbstractOptimizer
    
    properties
        LearnRate
        GradientThreshold
    end
    
    
    methods (Abstract)
        [W, cache] = updateWeight(optimizer, W, dLdW, cache);
        cache = initializeCache(optimizer, dLdW);
    end
    
    
    methods
        function optimizer = AbstractOptimizer(varargin)
            p = inputParser();
            p.KeepUnmatched = true;
            p.addParameter('LearnRate', 1e-2, @(x) isscalar(x) && x>0);
            p.addParameter('GradientThreshold', Inf, @(x) isscalar(x) && x>0);
            p.parse(varargin{:});
            opt = p.Results;
            
            optimizer.LearnRate         = opt.LearnRate;
            optimizer.GradientThreshold = opt.GradientThreshold;
        end
        
        
        function updateLayer(optimizer, layer, dLdParams)
            params = layer.getLearnableParameters(); % dparams{i} is the gradient of params{i}
            if isempty(params) % no parameter to update
                return
            end
            
            cache = layer.cache;
            if isempty(cache) %  initialization for the first time
                cache = cellfun(@optimizer.initializeCache, dLdParams, 'UniformOutput', false);
            end
            
            for i = 1:length(params) % update each parameter of given layer
                p = params{i};
                dLdW = gradientClip(dLdParams{i}, optimizer.GradientThreshold);
                [layer.(p), cache{i}] = optimizer.updateWeight(layer.(p), dLdW, cache{i});
            end
            
            layer.cache = cache;
        end
    end
end

function grad = gradientClip(grad, thres)
% gradientClip helps avoid NAN gradient
    grad = min(max(grad, -thres),thres);
end

