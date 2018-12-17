function [X,Y] = readData_iris()
% readData_iris reads iris dataset
%
% See also: readData
    dataurl    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
    datadir    = 'data';
    filename   = fullfile(datadir,'Iris.txt');
    
    if ~isfolder(datadir)
        mkdir(datadir);
    end
    if ~isfile(filename)
        websave(filename, dataurl);
    end

    assert(isfile(filename), [filename,' not found']);
    
    data = readtable(filename);
    X = table2array(data(:,1:4))';
    Y = table2cell(data(:,5)); % convert table to cell
    Y = cellfun(@labelclass, Y, 'UniformOutput', false); % convert string to vector
    Y = cell2mat(Y)'; % convert cell to matrix
    return 
    
    function num = labelclass(str) % onehot encoding
        switch str
            case 'Iris-setosa'
                num = [1,0,0];
            case 'Iris-versicolor'
                num = [0,1,0];
            case 'Iris-virginica'
                num = [0,0,1];
            otherwise
                error(['unrecognized iris class', str])
        end
    end
end