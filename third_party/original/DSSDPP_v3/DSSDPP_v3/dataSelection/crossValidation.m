function acc = crossValidation(data,label,v)
% data: N*dim
% label: 1*N
% if v = 10,it means 10-fold cross validation

acc = [];
step = floor(size(data,1)/v); % data: N * dim 
for k = 1:v
    if k ~= v
        startpoint = (k-1)*step+1;
        endpoint = k*step;
    else
        startpoint = (k-1)*step+1;
        endpoint = size(data,1);
    end
    cv_p = startpoint:endpoint; %%%% test set position

    %%%%%%%%%%%%%% test set
     test_data = data(cv_p,:); 
     test_label = label(cv_p);  
    %%%%%%%%%%%%%% training set
     train_data = data;
     train_data(cv_p,:) = [];       
     train_label = label;
     train_label(cv_p) = []; 
         
    % LR 
    model = train(train_label', sparse(train_data),'-s 0 -c 1 -B -1 -q'); % num * fec
    [predict_label, accuracy, prob_estimates] = predict(test_label', sparse(test_data), model, '-b 1');
%     h = prob_estimates(:,1)';
%     h = predict_label';
   
     acc = [acc,accuracy(1)];
end
acc = mean(acc);
