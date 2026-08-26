function [balancedDataset,Y] = MPOS(data,label)
% metric value permutation based over-sampling MPOS
% 

posData = data(:,label==1);
negData = data(:,label==0);
[d,posNum] = size(posData);
negNum = size(negData,2);

% generated minority samaple
T = negNum-posNum;
if T <= 0
    balancedDataset = data;
    Y = label;
    return
end

% sampled data
samData = zeros(d,T);
for i = 1:T
    X = zeros(d,1);
    for j = 1:d
        idx = randi(posNum);
        curX = posData(:,idx);
       
        D = pdist2(curX', posData');
        [dists, neighbors] = sort(sqrt(D'));
        
        r = rand(1);
        X(j) = curX(j)+r*(curX(j)-posData(j,neighbors(1)));
    end
    samData(:,i) = X;
end

balancedDataset = [samData,data];
Y = [ones(1,T),label];
