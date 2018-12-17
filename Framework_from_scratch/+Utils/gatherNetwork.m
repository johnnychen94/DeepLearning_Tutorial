function net = gatherNetwork(net)
% gpuNetwork move parameters of network from gpu to cpu
    assert(iscell(net), 'Network should be cell of layers');
    
    for i = 1:length(net) - 1
        net{i} = gatherLayer(net{i});
    end
end


function layer = gatherLayer(layer)
    params = layer.getLearnableParameters();
    for i = 1:length(params)
        layer.(params{i}) = gather(layer.(params{i}));
    end
end