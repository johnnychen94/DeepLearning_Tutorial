function LearnRates = LearnRateScheduler(InitialLearnRate, MaxEpoch, LearnRateDropPeriod, LearnRateDropFactor)
% LearnRateScheduler generates a sequence of learning rate using step decay method
    LearnRates = repmat(InitialLearnRate,MaxEpoch,1);
    numDrops = ceil(MaxEpoch/LearnRateDropPeriod);
    for i = 2:numDrops
        l = (i-1)*LearnRateDropPeriod + 1;
        r = min(i*LearnRateDropPeriod, MaxEpoch);
        LearnRates(l:r) = LearnRates(l-1) * LearnRateDropFactor;
    end
end