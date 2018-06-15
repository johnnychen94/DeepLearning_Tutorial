function [Xtrain,Ytrain,Xtest,Ytest] = load_house_price(a,noiselevel)
  Xtrain = 50 + 100*rand(200,1);
  Xtrain = Xtrain(Xtrain>50);
  Ytrain = -50 + 2 * Xtrain + a* Xtrain.*Xtrain;
  Ytrain = Ytrain + noiselevel*randn(size(Xtrain));
  
  Xtest = 50 + 100*rand(200,1);
  Xtest = Xtest(Xtest>50);
  Ytest = -50 + 2 * Xtest + a* Xtest.*Xtest;
  Ytest = Ytest + noiselevel*randn(size(Xtest));
  
end
