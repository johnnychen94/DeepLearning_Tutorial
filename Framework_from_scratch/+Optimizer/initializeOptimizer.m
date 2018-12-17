function optimizer = initializeOptimizer(optimizer, option)
%   initializeOptimizer 
    import Optimizer.*
    
    SupportedOptimizerList = ["sgdm",];
    validatestring(optimizer, SupportedOptimizerList, mfilename);
    
    optimizerList = struct(...
        "sgdm", @() SGDM(option.LearnRate, option.Momentum, option.GradientThreshold));
    
    optimizer_ = optimizerList.(optimizer); % function handle
    optimizer = optimizer_();
end