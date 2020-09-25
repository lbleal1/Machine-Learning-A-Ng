function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Return the following variables correctly.
idx = zeros(size(X,1), 1); % 15 x1

for i=1:size(X,1),
  pos_idx = zeros(K,1); 
  for j=1:K,
    pos_idx(j) = sqrt(sum((X(i,:) - centroids(j,:)).^2)); 
  [minval, idx(i)] = min(pos_idx);
end

end

