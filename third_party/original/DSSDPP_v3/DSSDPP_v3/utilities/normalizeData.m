function [X,Y] = normalizeData(data)

X = data(1:end-1,:);
Y = data(end,:);
Y(Y>1) = 1;

% normaliztion 
X = zscore(X,0,2); 

