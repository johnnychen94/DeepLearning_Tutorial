function Y = classify(net, X)
    assert(iscell(net), 'Network should be cell of layers');
    
    import Utils.predict
    
    [~,Y] = max(predict(net,X));
end