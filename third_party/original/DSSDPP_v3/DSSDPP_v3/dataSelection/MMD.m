function dist = MMD(Xs,Xt)
% Maximum Mean Discrepancy  
% 
% calculate the distance between source Xs and target Xt
% 

ns = size(Xs,2);
nt = size(Xt,2);

% calculate kernel matrix
sigma = median(pdist([Xs, Xt]'));

Ks = constructKernel(Xs',[],sigma);
Kt = constructKernel(Xt',[],sigma);
Kst = constructKernel(Xs',Xt',sigma);

c_Ks = 1/(ns^2);
c_Kt = 1/(nt^2);
c_Kst = 2/(ns*nt);

dist = sum(sum(c_Ks.*Ks))+sum(sum(c_Kt.*Kt))-sum(sum(c_Kst.*Kst));
   
end


function K = constructKernel(X,Y,sigma)
[nx,~] = size(X);
sqX = sum((X.*X),2);

if isempty(Y)
    Q = repmat(sqX,1,nx);
    D = Q + Q' - 2*(X*X'); 
else
    [ny,~] = size(Y);
    sqY = sum((Y.*Y),2);
    Q = repmat(sqX,1,ny);
    R = repmat(sqY',nx,1);
    D = Q + R - 2*(X*Y'); 
end

K = exp(-D/(2*sigma^2));  

end
