function dist = DCNNS(Xs,Xt)

ds = size(Xs,1);
dt = size(Xt,1);

XsDCV = [];
for is =1:ds
    XsDCV = [XsDCV; getDCV(Xs(is,:))];
end

XtDCV = [];
for it=1:dt
    XtDCV = [XtDCV; getDCV(Xt(it,:))];
end

dist = pdist2(XsDCV',XtDCV', 'euclidean');


function [DCV] = getDCV(vector)
% compute the distributional characteristics of a sample or feature 
%  mode, median, mean, harmonic mean, minimum, maximum, range, variation
%  ratio, first quaritle, third quartile, interquartile range, variance,
%  standard devariation, coefficient of variation, skewness, kurtosis
% 
%  ASE2012_an investigation on the feasibility of cross-project defect
%  prediction -- Table 7
% 
% vector: n X 1

DCV = zeros(5,1);
DCV(1) = mean(vector);
DCV(2) = std(vector);
DCV(3) = median(vector);
DCV(4) = min(vector);
DCV(5) = max(vector);


