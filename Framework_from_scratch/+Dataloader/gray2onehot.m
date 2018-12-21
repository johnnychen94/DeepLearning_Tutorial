function onehot = gray2onehot(gray, n)
% gray2onehot converts gray encoding to onehot encoding
%
% Examples:
%   for exact gray encoding:
%       1 --> [1,0,0]
%       2 --> [0,1,0]
%       3 --> [0,1,1]
% See also: onehot2gray    
    assert(isvector(gray));
    
    onehot = zeros(n, length(gray));
    gray = gray - min(gray) + 1;
    
    for i = 1:length(gray)
        onehot(gray(i),i) = 1;
    end
end