function update(optimizer, net, dLdParams)
    assert(iscell(net), 'Network should be cell of layers');
    
    for i = 1:length(net)-1
        optimizer.update(net{i}, dLdParams{i});
    end
end