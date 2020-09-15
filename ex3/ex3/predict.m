function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1); % variable to return
 
a_1 = [ones(m, 1) X];

z_2 = a_1*Theta1'; % result is 16x4
a_2 = sigmoid(z_2);

a_2 = [ones(size(a_2,1),1) a_2]

z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);

[max_h, p] = max(a_3, [], 2)

end
