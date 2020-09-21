function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% cost function
% X: 10x3 | y: 10x1 | theta: 3x1 | lambda: 1x1 | grad: 9x1
J = (1/(2*m))*sum((X*theta - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

% =========================================================================
grad(1) = ((1/m)*(X*theta- y)'*X(:,1));
grad(2:end) = (1/m)*((X*theta-y)'*X(:,2:end))' + (lambda/m)*theta(2:end);
grad = grad(:);
end
