classdef (Abstract) AbstractOptimizer
    methods (Abstract)
        update(optimizer, layer, dLdParams);
    end
    
    methods
        function cache = initializeCache(~, dLdParams)
            % initialize cache with all zeros
            cache = cell(size(dLdParams));
            for i = 1:length(cache)
                cache{i} = zeros(size(dLdParams{i}), 'like', dLdParams{i});
            end
        end
    end
end

