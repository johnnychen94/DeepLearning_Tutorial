function rst = accuracy(Y, Y_hat)
% accuracy returns classification accuracy
    rst = 1 - sum(Y_hat ~= onehot2gray(Y))/numel(Y);
end