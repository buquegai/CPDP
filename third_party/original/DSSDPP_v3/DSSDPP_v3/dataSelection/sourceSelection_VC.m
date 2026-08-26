clear 
close all
clc

addpath('..\utilities\')
addpath('.\liblinear\')

% load data
load('..\DATASET_50.mat')

% setting paramters
ratio = 0.1; % use 10% of target modules to select source data
rep = 30; % repeat 30 times

str = '..\output\';
if exist(str,'dir') == 0
    mkdir(str);
end

ALLDIST = cell(size(DATA,1),1);

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
     
    DIST = zeros(size(Xs,1)-1,rep);
    id = [];
    for loop = 1:rep
        dist = [];
        
        % select and normalize target data
        [Xt,Yt] = spiltData(target,IDX(loop,:),ratio);
        
        for j = 1:size(Xs,1)
            src_filen = Xs{j,2};
            if strcmp(tar_filen,src_filen) == 0
                source = Xs{j,1}.data;             
                [Xss,Ys] = normalizeData(source); % get and normlize source data
                                
                d = VC(Xss,Xt);
                dist = [dist; abs(d)];
                
                if loop == rep
                    id = [id; j];
                end
            end % if
        end
        DIST(:,loop) = dist;
    end
   ALLDIST{i,1} = DIST;
   ALLDIST{i,2} = id;
end

[ID,numSources] = getIDbyDist(ALLDIST,'VC');
save([str,'VC_ID.mat'], 'ID','numSources')

disp('running programe done !')
