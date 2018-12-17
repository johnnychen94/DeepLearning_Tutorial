classdef MSERegressionLayer < Layer.AbstractLossLayer
% MSERegressionLayer returns a loss layer with Mean-Squared-Error
    methods
        function loss = forwardLoss(~,Y,T)
            assert(ismatrix(Y) && ismatrix(T), "%s accepts matrix input only", mfilename)
            
            squares = 0.5*(Y-T).^2;
            loss = mean(sum(squares,1));
        end

        function dX = backwardLoss(~,Y,T)
            numSample = size(Y,2);
            dX = (Y - T)/numSample;
        end
    end
end
