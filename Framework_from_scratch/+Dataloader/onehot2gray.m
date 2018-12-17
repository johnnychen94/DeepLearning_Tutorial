function gray = onehot2gray(onehot)
% onehot2gray converts onehot encoding to gray encoding
%
% This function is particularly useful for multi-class classification:
%   * convert onehot-encoding training labels to gray-encoding
%   * convert onehot-encoding network output to gray-encoding for easy visualization.
%
% Examples:
%   for exact onehot encoding:
%       [1,0,0] --> 1
%       [0,1,0] --> 2
%       [0,1,1] --> 3
%   for predicted onehot encoding:
%       [0.9, 0.1, 0.3] --> 0.9
%       [0.1, 0.9, 0.3] --> 1.9
%       [0.1, 0.3, 0.9] --> 2.9
%
% See also: gray2onehot
    onehot(onehot<0) = 0;
    onehot(onehot>1) = 1;
    
    [v,i] = max(onehot);
    gray = v + i - 1;
end