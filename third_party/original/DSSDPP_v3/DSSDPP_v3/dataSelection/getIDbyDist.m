function [ID,numSources] = getIDbyDist(DIST,opt)
% 
proNum = size(DIST,1) ;
tempDIST = cell(proNum,3);
ID = cell(proNum,1);
numSources = zeros(proNum,1);

switch opt % NASA-AEEEM-ReLink-Eclipse, use 10% target data
    case 'MMD'
        th = [repmat(0.025,1,11), repmat(0.01,1,5), repmat(0.02,1,3), repmat(0.01,1,3)]; % MMD
    case 'DCNNS'
        th = [repmat(40,1,11), repmat(60,1,5), repmat(20,1,3), repmat(100,1,3)]; % DCNNS
    case 'VC'
        th = [repmat(90,1,11), repmat(95,1,5), repmat(80,1,3), repmat(70,1,3)]; % VC
    case 'CORR'
        th = [repmat(0.2,1,11), repmat(0.2,1,5), repmat(0.6,1,3), repmat(0.025,1,3)]; % CORR
end

for i = 1:proNum 
    currentDist = DIST{i,1};
    tempDIST{i,1} = currentDist;
    
    meanD = mean(currentDist,2);
    tempDIST{i,2} = meanD;
    medianD = median(currentDist,2);
    tempDIST{i,3} = medianD;
    
    [smeanD,idx] = sort(meanD); % acesend
    ID{i,1} = DIST{i,2}(idx);
    
    c = length(find(smeanD<=th(i)));
    if c==0
        c = 1;
    end
    numSources(i) = c; %  number of selected sources by threshold
end



