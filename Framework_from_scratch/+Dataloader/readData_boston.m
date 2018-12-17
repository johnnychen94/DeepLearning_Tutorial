function [X,Y] = readData_boston()
% readData_Boston reads boston housing dataset
%
% See also: readData
    dataurl    = 'http://lib.stat.cmu.edu/datasets/boston';
    datadir    = 'data';
    filename   = fullfile(datadir,'BostonHousing.txt');
    
    if ~isfolder(datadir)
        mkdir(datadir);
    end
    if ~isfile(filename)
        websave(filename, dataurl);
    end

    assert(isfile(filename), [filename,' not found']);
    try 
        fileID = fopen(filename);
        data = parsedata(fileID);
        X = data(:,1:end-1)';
        Y = data(:,end)';
        fclose(fileID);
    catch err
        fclose(fileID);
        rethrow(err)
    end
    
    [X,Y] = sortdata(X,Y);
end

function data = parsedata(fileID)
    row_offset  = 23; % records start from line 23
    num_records = 506;
    num_columns = 14;

    % pass description
    for i = 1: row_offset - 1
        fgetl(fileID);
    end

    data = zeros(num_records,num_columns);
    i = 1;
    while i <= 506
        % each record contains two lines of file
        part1 = fgetl(fileID);
        part2 = fgetl(fileID);

        record = textscan([part1,part2],'%f'); % cell
        data(i,:) = record{:};
        i = i + 1;
    end
end

function [sorted_X,sorted_Y] = sortdata(X,Y)
% sortdata sorts [X,Y] in order of Y
    [sorted_Y, I] = sort(Y);
    sorted_X = X(:,I);
end