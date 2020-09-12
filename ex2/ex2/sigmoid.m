function g = sigmoid(z)

% Initialize the variable to be returned
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
g = 1./(1+e.^(-z))
% =============================================================

end

