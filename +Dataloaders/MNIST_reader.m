% filename: MNIST_reader.m
function [imgs,labels] = MNIST_reader(root,varargin)
  % Read MNIST dataset images and labels in one time
  %
  % Return:
  %   imgs :  4-D numeric array [28,28,1,60000](train) or [28,28,1,10000](test)
  %   labels : categorical vector of labels
  % Parameters:
  %   root(Required): the root path folder of MNIST dataset
  %   train(Optional):  true(default) | false, load train dataset if train == true
  p = inputParser();
  p.addParameter('train',true,@islogical);
  p.parse(varargin{:});
  
  % get image and label filename
  if p.Results.train == true
    img_file = dir(fullfile(root,'*train*image*'));
    label_file = dir(fullfile(root,'*train*label*'));
  else
    img_file = dir(fullfile(root,'*t10k*image*'));
    label_file = dir(fullfile(root,'*t10k*label*'));
  end
  img_file = fullfile(img_file.folder,img_file.name);
  label_file = fullfile(label_file.folder,label_file.name);
  
  % read image and label
  imgs = image_reader(img_file);
  labels = label_reader(label_file);
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
  %   labels : categorical vector of labels
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
%     char*1 is 1 unsigned byte
    labels{i}  = num2str(uint8(fread(fID,1,'char*1')));
  end
  catch e
    fclose(fID);
    rethrow(e);
  end
  labels = categorical(labels);
  fclose(fID);
end