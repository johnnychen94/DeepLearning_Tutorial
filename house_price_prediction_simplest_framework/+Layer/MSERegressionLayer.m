classdef MSERegressionLayer < handle
  
  methods
    function layer = MSERegressionLayer()
    end
    
    function loss = forwardLoss(layer,Y,T)
      squares = 0.5*(Y-T).^2;
      loss = mean(squares);
    end
    
    function dX = backwardLoss(layer,Y,T)
      dX = (Y - T)/numel(Y);
    end
  end
end
