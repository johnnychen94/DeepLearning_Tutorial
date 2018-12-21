function Y = batch_imresize(X, sz)
    assert(isnumeric(X) && length(size(X)) == 4);
    assert(isnumeric(sz) && length(size(sz)) == 2);
    
    Y = zeros([reshape(sz(:),1,numel(sz)), size(X,3), size(X,4)], 'like', X);
    for i = 1: size(X,4)
        % TODO: this for-loop might be unnecessary by implementing a 4-D version imresize
        Y(:,:,:,i) = imresize(X(:,:,:,i), sz);
    end
end