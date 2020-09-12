function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% Return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
size_t= size(theta);

h = sigmoid(X*theta);
J = (1/m)*(sum( ( (-y)'*log(h) ) - (1-y)'*(log(1-h)) )) + (lambda/(2*m))*sum((theta(2:size_t)).^2);

grad_theta=zeros(size_t);
grad_theta(2:size_t)=(lambda/m)*(theta(2:size_t));
grad = (1/m)*((X')*(h - y)) + grad_theta;
% =============================================================
end
