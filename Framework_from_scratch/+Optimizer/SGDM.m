classdef SGDM < Optimizer.AbstractOptimizer
% SGDM is stocastic gradient descent optimizer with momentum
    properties
        Momentum
    end
    
    methods
        function optimizer = SGDM(varargin)
        % SGDM returns a SGDM (stochastic gradient descent with momentum) optimizer
        %
        % Input:
        %   LearnRate : (Parameter)
        %       Default: 1e-2
        %   Momentum  : (Parameter)
        %       Default: 0.9
        %   GradientThreshold : (Parameter)
        %       threshold for gradient clipping
        %       Default: Inf
            [opt, unmatched] = parseInput(varargin{:});
            optimizer = optimizer@Optimizer.AbstractOptimizer(unmatched{:});
            optimizer.Momentum = opt.Momentum;
        end
        
        function [W, cache] = updateWeight(optimizer, W, dLdW, cache)
        % updateWeight uses nesterov momentum method
            cache_prev = cache;
            cache = optimizer.Momentum * cache - optimizer.LearnRate * dLdW;
            W = W - optimizer.Momentum * cache_prev + (1+optimizer.Momentum) * cache;
        end
        
        function cache = initializeCache(~, dLdW)
            cache = zeros(size(dLdW), 'like', dLdW);
        end
    end
end

function [opt, unmatched] = parseInput(varargin)
    p = inputParser();
    p.KeepUnmatched = true;

    p.addParameter('Momentum', 0.9, @(x) isscalar(x) && x>0);

    p.parse(varargin{:});
    opt = p.Results;
    unmatched = [fieldnames(p.Unmatched), struct2cell(p.Unmatched)]';
end
