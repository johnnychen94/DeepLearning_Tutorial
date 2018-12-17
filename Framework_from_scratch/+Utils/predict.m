function Y = predict(net, X)
    assert(iscell(net), 'Network should be cell of layers');
    
    Y = X;
    for i = 1:length(net)-1
        Y = net{i}.predict(Y);
    end
end