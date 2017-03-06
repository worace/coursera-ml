function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

# Theta1
# 25 x 401 matrix for first layer weights
# 401 inputs -> 25 hidden layer nodes
# Theta2
# 10 x 26 matrix for output layer weights
# 26 inputs (25 hidden nodes + 1 bias unit in hidden layer)
# x 10 output nodes (one for each class, i.e. digits 0 - 9)

# X - input matrix
# 5000 x 400 matrix, 5000 examples x 400 inputs (pixels)

hidden_layer = sigmoid([ones(size(X,1), 1) X] * Theta1');
p = predictOneVsAll(Theta2, hidden_layer);

# Could do this prediction ourselves by generating output layer
# and then calculating the max index from it, but since we already
# wrote the predictOneVsAll function we can just use that for the
# hidden layer -> output layer translation
## output_layer = sigmoid([ones(size(hidden_layer, 1), 1) hidden_layer] * Theta2');

% =========================================================================
end
