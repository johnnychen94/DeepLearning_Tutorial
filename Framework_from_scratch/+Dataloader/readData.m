function [X,Y] = readData(dataset, varargin)
% readData returns training data
% 
% Usage:
%   [X,Y] = readData(dataset)
%   [X,Y] = readData(dataset, varargin)
%
% Input:
%   dataset : (Required) String
%       Current supported datasets are:
%           'toy'    : toy dataset for hw3
%           'boston' : The Boston Housing Dataset
%           'iris'   : Iris DataSet
%           'mnist'  : MNIST Dataset
%   varargin : (Optional)
%       Any other arguments are passed into the actual dataReader function
%
% Return:
%    X : Array
%       input data
%    Y : Array
%       label/groundtruth of input data
% Note:
%   * By convention, the last dimension indicates the index of sample, 
%     i.e., X(:,1) stands for the first sample.
    import Dataloader.*
    
    dataset = lower(dataset);
    
    SupportedDatasetList = ["iris", "mnist", "boston", "toy"];
    validatestring(dataset, SupportedDatasetList, mfilename);
    
    readerLists = struct(...
        'toy',@readData_toy,...
        'boston',@readData_boston,...
        'iris',@readData_iris,...
        'mnist',@readData_mnist );
    
    readData_ = readerLists.(dataset);
    [X,Y] = readData_(varargin{:});
end