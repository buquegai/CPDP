function dist = VC(Xs,Xt)
K = 500;
ns = size(Xs,2);
nt = size(Xt,2);

n = min(ns,nt);
K = min(K,n);

idxs = randperm(ns);
idxt = randperm(nt);
Xss = Xs(:,idxs(1:K));
Xtt = Xt(:,idxt(1:K));

X = [Xss,Xtt];
Y = [-1*ones(1,K),ones(1,K)];

acc = crossValidation(X',Y,5);

dist = 2*(acc-0.5);
