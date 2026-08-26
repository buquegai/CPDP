function score  = DSSDPP(Xs,Ys,Xt,lp)
% domain programming predictor based easy transfer learning

% Inputs:
%%% Xs          : source data, ns * m
%%% Ys          : source label, ns * 1
%%% Xt          : target data, nt * m
%%%%%% The following inputs are not necessary
%%% lp          : linear(default)|binary

% Outputs:
%%% score      : predictions for target data
    
% Reference:
% Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang.
% Easy Transfer Learning By Exploiting Intra-domain Structures.
% IEEE International Conference on Multimedia & Expo (ICME) 2019.

    C = length(unique(Ys));  % num of classes
    if C > max(Ys)
        Ys = Ys + 1;
    end
    m = size(Xt,1); % number of modules in target project 

    if nargin == 3
        lp = 'linear';
    end
   
    [~,Dct]= get_class_center(Xs,Ys,Xt);
    
    % domain programming... 
    [Mcj] = DPP(C,m,Dct,lp);
    score = Mcj(:,2);
    score = score';
end


function [source_class_center,Dct] = get_class_center(Xs,Ys,Xt)
% Get source class center and Dct
    source_class_center = [];
    Dct = [];
    class_set = unique(Ys)';
     
    for i = class_set
        indx = Ys == i;
        X_i = Xs(indx,:);
        
        % find class center
        meanPoint = mean(X_i);
        source_class_center = [source_class_center,meanPoint'];
        
        % standard Euclidean distance
        S = nanstd(Xt);
        S(S==0) = 1;
        Dct_c = pdist2(Xt,meanPoint,'seuclidean',S);
        
        Dct = [Dct, Dct_c];
    end
end
