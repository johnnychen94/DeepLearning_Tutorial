function [imgs,labels] = readData_mnist(varargin)
% Read MNIST dataset images and labels in one time
% see http://yann.lecun.com/exdb/mnist/ for file format details
%
% Usage:
%   [X,Y] = readData_mnist();
%   [X,Y] = readData_mnist(set);
%
% Return:
%   imgs :  4-D numeric array [28,28,1,60000](train) or [28,28,1,10000](test)
%   labels : categorical vector of labels
%
% Parameters:
%   set : (Optional) String
%     * "train" (default)
%         returns training set
%     * "test"
%         returns test set
%
% See also: readData
  root = "http://yann.lecun.com/exdb/mnist/";
  datadir = fullfile('data','mnist');
  fileList = ["train-images-idx3-ubyte"
      "train-labels-idx1-ubyte"
      "t10k-images-idx3-ubyte"
      "t10k-labels-idx1-ubyte"];
  urlList  = strcat(root, fileList, '.gz');
  fileList = fullfile(datadir, fileList);
  
  set = parseInput(varargin{:});
  
  if ~isfolder(datadir)
      mkdir(datadir);
  end
  
  arrayfun(@(url, file) download_extract(url, file, datadir), urlList, fileList);
  
  % get image and label filename
  if strcmp(set, 'train')
    imgFile = fileList(1);
    labelFile = fileList(2);
  else
    imgFile = fileList(3);
    labelFile = fileList(4);
  end
  
  % read image and label
  imgs = image_reader(imgFile);
  labels = label_reader(labelFile);
end

function download_extract(url, filename, datadir)
    if ~isfile(filename)
        gunzip(url, datadir);
    end
end

function imgs = image_reader(file)
% Read MNIST image from binary file
% Return:
%   imgs :  4-D numeric array [28,28,1,60000](train) or [28,28,1,10000](test)
% Parameters:
%   file(Required) : full filepath of binary file of MNIST image
%
% File Formats:
% IMAGE FILE (*-idx3-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000803(2051) magic number 
% 0004     32 bit integer  60000            number of images 
% 0008     32 bit integer  28               number of rows 
% 0012     32 bit integer  28               number of columns 
% 0016     unsigned byte   ??               pixel 
% 0017     unsigned byte   ??               pixel 
% ........ 
% xxxx     unsigned byte   ??               pixel
%
% Pixels are organized row-wise. 
% Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
  try 
    fID = fopen(file,'rb');frewind(fID);

    % read and check file information
    magic_number = fread(fID,1,'int','b');
    img_num = int32(fread(fID,1,'int','b'));
    img_height = int32(fread(fID,1,'int','b')); 
    img_width = int32(fread(fID,1,'int','b'));
    if ~((magic_number ==2051) && ...
        (img_num == 60000 || img_num == 10000) && ...
        (img_height == 28) && (img_width == 28))
        error(['Invalid MNIST Image file: ',file]);
    end

    % read images
    imgs = zeros(img_height,img_width,1,img_num);
    for i = 1:img_num
      % uint8 is 1 unsigned byte
      temp_img = uint8(fread(fID,img_height*img_width,'uint8','b'));
      imgs(:,:,1,i) = 255-reshape(temp_img,[img_height,img_width])';
    end
    
  catch e
    fclose(fID);
    rethrow(e)
  end
  
  fclose(fID);
end

function labels = label_reader(file)
% Read MNIST label from binary file
% Return:
%   labels : 2-D array
% Parameters:
%   file(Required) : full filepath of binary file of MNIST label
%
% LABEL FILE (*-idx1-ubyte):
% 
% [offset] [type]          [value]          [description] 
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
% 0004     32 bit integer  10000            number of items 
% 0008     unsigned byte   ??               label 
% 0009     unsigned byte   ??               label 
% ........ 
% xxxx     unsigned byte   ??               label
% The labels values are 0 to 9.
  try 
    fID = fopen(file,'rb');frewind(fID);
    % read and check file information
    magic_number = fread(fID,1,'int','b');
    label_num = int32(fread(fID,1,'int','b'));
    if ~((magic_number ==2049) && ...
        (label_num == 60000 || label_num == 10000))
      error(['Invalid MNIST Label file: ',file]);
    end

    % read image labels
    labels = cell(label_num,1);
    for i = 1:label_num
        % char*1 is 1 unsigned byte
        labels{i}  = num2str(uint8(fread(fID,1,'char*1')));
    end
  catch e
    fclose(fID);
    rethrow(e);
  end
  
  fclose(fID);
  labels = cellfun(@str2double, labels)';
end

function set = parseInput(varargin)
    P = inputParser();
    P.addOptional('set','train', @ischar);
    P.parse(varargin{:});
    set = P.Results.set;

    validSetList = ["train", "test"];
    set = validatestring(set, validSetList, mfilename);
end