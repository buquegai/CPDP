clear 
close all
clc

addpath('.\utilities\')
addpath(genpath('.\featureselection\'))

% Load target
load('.\DATASET_50.mat')
load('.\output\CORR_ID.mat')

% default paramters setting 
ratio = 0.9; % select 90% of target modules as test set
rep = 30; % repeat 30 times
time = zeros(size(DATA,1),rep); % running time
expRESULT = cell(size(DATA,1),1); % experimental results

methodName = 'mDSSDPP'; 
str = '.\output\';
if exist(str,'dir') == 0
    mkdir(str);
end
savePath = [str,methodName];

for i = 1:size(DATA,1)  % dataset
    target = DATA{i,1}.data;
    tar_filen = DATA{i,2};% project name
    IDX = DATA{i,3}; % random index
 
    if i<=11 % NASA 11
        Xs = DATA(1:11,:);
    elseif i>11 && i<=16 % AEEEM 5
        Xs = DATA(12:16,:);
    elseif i>16 && i<=19 % ReLink 3 
        Xs = DATA(17:19,:);
    else % Eclipse 3 
        Xs = DATA(20:end,:);
    end
    
    % the index of selected source, the selected mulitple similar sources 
    sID = ID{i,1}(1:numSources(i)); 
    Xs = Xs(sID,1);
    
    source = [];
    for j = 1:size(Xs,1)
        source = [source,Xs{j,1}.data];
    end       
    [Xss,Ys] = getData(source); % get source data

    % MPOS algorithm
    rng('default')
    [Xss,Ys] = MPOS(Xss,Ys);
    
    % metric selection
    yy = Ys+1;
    out = fsCFS(Xss',yy');
    feaId = out.fList;
    Xss = Xss(feaId,:);

    Xss = zscore(Xss,0,2);  % normalize the source data
    
    MEASURE = cell(rep,1);
    t = zeros(1,rep); % time
    for loop = 1:rep
        tStart = tic; % timer start 
        
        % select and normalize target data
        [Xt,Yt] = spiltData(target,IDX(loop,:),ratio);
        Xt = Xt(feaId,:);

        % running DSSDPP method
        score = DSSDPP(Xss',Ys',Xt');

        % calculate measure
        measure = performanceMeasure(Yt,score); 

        % timer end
        tElapsed = toc(tStart);
        t(1,loop)  = tElapsed;
        MEASURE{loop,1} = measure;
    end
    time(i,:) = t;
    expRESULT{i,1} = MEASURE;
end

% calcuate running time 
totalT = sum(time,1);
meanT = sprintf('%0.3f',mean(totalT));
meanT = strcat(meanT,'¡À');
stdT = sprintf('%0.3f',std(totalT));
meanstdT = strcat(meanT, stdT);

% computer performance measures
result = getExpResults(expRESULT);

save([savePath,'_EXPRESULT.mat'], 'expRESULT','result','time','meanstdT')
disp('running programe done !')
