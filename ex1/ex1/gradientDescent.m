function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
  theta = (theta) - ((1/m)*((X*theta)-y)'*X)'*alpha

% m: 1  x 1
% X: 97 x 1
% theta: 2 x 1
% y: 97 x 1
% alpha: 1 x 1
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

