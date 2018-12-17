function options = trainingOptions(varargin) % variable-length input argument list
    P = inputParser();
    
    P.addParameter('Optimizer', 'sgdm', @(x) ischar(x));
    P.addParameter('InitialLearnRate', 1e-2, @(x) isscalar(x) & x>0);
    P.addParameter('LearnRateDropFactor', 0.1, @(x) isscalar(x) & x>0);
    P.addParameter('Momentum', 0.9, @(x) isscalar(x) & x>0);
    P.addParameter('GradientThreshold', Inf, @(x) isscalar(x) & x > 0);
    
    P.addParameter('BatchSize', 128, @(x) mod(x,1)==0 & isscalar(x) & x > 0);
    P.addParameter('MaxEpoch', 50, @(x) mod(x,1)==0 & isscalar(x) & x > 0);
    P.addParameter('LearnRateDropPeriod', 20, @(x) mod(x,1)==0 & isscalar(x) & x > 0);
    
    P.addParameter('UseGPU', false, @(x) islogical(x));
    P.addParameter('Shuffle', true, @(x) islogical(x));
    P.addParameter('Verbose', false, @(x) islogical(x));
    P.addParameter('VerboseFrequency', 100, @(x) mod(x,1)==0 & isscalar(x) & x > 0);
    
    P.parse(varargin{:});
    
    options = P.Results;
end