classdef CrossEntropyLayer < Layer.AbstractLossLayer
    
    methods
        function loss = forwardLoss( ~, Y, T )
            % forwardLoss    Return the cross entropy loss between estimate
            % and true responses averaged by the number of observations
            
            assert(ismatrix(Y) && ismatrix(T), "%s accepts matrix input only", mfilename)
            
            numElems = size(Y,2);
            loss = -sum( sum(T.* log(Layer.Utils.boundAwayFromZero(Y))) ./numElems );
        end
        
        
        function dX = backwardLoss( ~, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            
            numElems = size(Y,2);
            dX = (-T./Layer.Utils.boundAwayFromZero(Y)) /numElems;
        end
    end
end
