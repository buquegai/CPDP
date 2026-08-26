function [X,Y] = getData(data)

X = data(1:end-1,:);
Y = data(end,:);
Y(Y>1) = 1;


