function Dist = CORR(Xs,Ys,Xt)

dim = size(Xs,1);
dist = [];
for i = 1:dim
    d = pdist2(Xs(i,:),Ys,'spearman');
    dist = [dist;d];
end
dist(isnan(dist)) = 0;

[~,idx] = sort(dist,'descend');
q = Xs(idx(1:3),:);
r = Xt(idx(1:3),:);

Q = zeros(size(q,1),1);
for i = 1:size(q,1)
    for j = 1:size(q,1)
        if i~=j
            Q(i) = pdist2(q(i,:),q(j,:),'spearman');
        end
    end
end

R = zeros(size(r,1),1);
for i = 1:size(r,1)
    for j = 1:size(r,1)
        if i~=j
            R(i) = pdist2(r(i,:),r(j,:),'spearman');
        end
    end
end

Dist = pdist2(Q',R');
