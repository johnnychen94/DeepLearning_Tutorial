function net = gpuNetwork(net)
% gpuNetwork move parameters of network from cpu to gpu
    assert(iscell(net), 'Network should be cell of layers');
    
    for i = 1:length(net) - 1
        net{i} = gpuLayer(net{i});
    end
end


function layer = gpuLayer(layer)
    params = layer.getLearnableParameters();
    for i = 1:length(params)
        layer.(params{i}) = gpuArray(layer.(params{i}));
    end
end