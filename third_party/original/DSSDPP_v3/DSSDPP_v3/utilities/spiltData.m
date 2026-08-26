function [X,Y] = spiltData(data,idx,ratio)
% split target data
% 

n = size(data,2); % the number of target modules

% random index
testIdx = idx(1:floor(ratio*n));
% trIdx = setdiff(idx,testIdx,'stable');

% test target data
X = data(1:end-1,testIdx);
Y = data(end,testIdx); % label
Y(Y>1) = 1;

% zscore normalization
X = zscore(X,0,2);










