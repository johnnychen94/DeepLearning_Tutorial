function optimizer = initializeOptimizer(optimizer, option)
%   initializeOptimizer 
    import Optimizer.*
    option = [fieldnames(option), struct2cell(option)]'; % convert struct to varargin
    
    SupportedOptimizerList = ["sgdm",];
    validatestring(optimizer, SupportedOptimizerList, mfilename);
    
    optimizerList = struct(...
        "sgdm", @() SGDM(option{:}));
    
    optimizer_ = optimizerList.(optimizer); % function handle
    optimizer = optimizer_();
end