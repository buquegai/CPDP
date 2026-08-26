function result = getExpResults(RESULT)
% calculate the experimental results

proNum = size(RESULT,1) ;
tempRESULT = cell(proNum,4);
for i = 1:proNum 
    currentRes = cell2mat(RESULT{i,1});
    tempRESULT{i,1} = currentRes;
    tempRESULT{i,2} = mean(currentRes,1);
    tempRESULT{i,3} = median(currentRes,1);
    tempRESULT{i,4} = std(currentRes,1);
end

%% statistic results
all_ONE = [];  mean_ONE = []; median_ONE = []; 
std_ONE = [];

% projectN = size(tempRESULT,1);
for i = 1:proNum
    all_ONE = [all_ONE; tempRESULT{i,1}];
    mean_ONE = [mean_ONE; tempRESULT{i,2}];
    median_ONE = [median_ONE; tempRESULT{i,3}];   
   
    std_ONE = [std_ONE; tempRESULT{i,4}];
end

meanAll_ONE = mean(all_ONE,1); 
stdAll_ONE = std(all_ONE,1); 
medianAll_ONE = median(all_ONE,1);

mean_ONE = [mean_ONE; meanAll_ONE]; 
std_ONE = [std_ONE; stdAll_ONE];
median_ONE = [median_ONE; medianAll_ONE];

%% save to file
result = cell(1,5);
result{1,1} = mean_ONE; result{1,2} = median_ONE; 
result{1,3} = all_ONE; result{1,4} = tempRESULT; 
result{1,5} = std_ONE; 
