function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer %neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% input_layer_size: 400 (individual pixel features from images)
% hidden_layer_size: 25
#
#
#
# Theta1
# 401 x 25 matrix for a2 weights
# Theta2
# 26 x 10 matrix for a3 (output) weights
# note when fed to the function these seem to already
# be transposed (I guess to make multiplication easier)
#
# input_layer_size = 400
# hidden_layer_size = 25
# (not counting bias params)
#
# nn_params : 10285 x 1 vector of unrolled
# Theta1 and Theta2 concatenated
# nn_params -> Theta1;
# Theta1 is 401x25 = 10025 units long
# where 401 = input layer size and
# 25 = hidden layer size

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
  theta1_length = hidden_layer_size * (input_layer_size + 1);
  theta1_vec = nn_params(1:theta1_length);
  Theta1 = reshape(theta1_vec,
                   hidden_layer_size,
                   (input_layer_size + 1));

  theta2_length = (hidden_layer_size + 1) * num_labels;
  theta2_vec = nn_params((theta1_length + 1):theta1_length + theta2_length);
  Theta2 = reshape(theta2_vec,
                   num_labels,
                   (hidden_layer_size + 1));

  m = size(X, 1);

  J = 0;

# Non-regularized Cost
# K = 10 (number of classes)
# m = 5000 (number of examples)
# cost = -1/m * sum over m ( sum over k (yki * log(h(xi))k + (1 - yki) * log(1 - h(xi))k ))
#
# 1. Get a2 from a1
# 2. Get a3 (output) from a2


  a1 = [ones(size(X,1),1) X];
  z2 = (a1 * Theta1');
  a2 = [ones(size(z2,1),1) sigmoid(z2)];
  z3 = (a2 * Theta2');
  a3 = sigmoid(z3);

  h_theta = a3;

  [confidences, classes] = max(h_theta, [], 2);

  bucketed_y = zeros(m, num_labels);

# Turn our list of classes (y) into a matrix
# of 1/0 flags for each class, i.e.
# 2
# 7
# ->
# 0 1 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 7 0 0 0

  for i = 1:size(X,1)
    class = y(i);
    bucketed_y(i, class) = 1;
  end

  sum = 0;
  for i = 1:size(X,1)
    for k = 1:num_labels
      pred = bucketed_y(i,k);
      h = h_theta(i,k);
      cost_per = pred * log(h) + (1 - pred) * log(1 - h);
      sum += cost_per;
    end
  end
  ## 0.287629
  J = - sum/m;



  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
% variable J. After implementing Part 1, you can verify that your
% cost function computation is correct by verifying the cost
% computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad. You should return the partial derivatives of
% the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% Theta2_grad, respectively. After implementing Part 2, you can check
% that your implementation is correct by running checkNNGradients
%
% Note: The vector y passed into the function is a vector of labels
%       containing values from 1..K. You need to map this vector into a
%       binary vector of 1's and 0's to be used with the neural network
%       cost function.
%
% Hint: We recommend implementing backpropagation using a for-loop
%       over the training examples if you are implementing it for the
%       first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%Hint: You can implement this around the code for
%      backpropagation. That is, you can compute the gradients for
%      the regularization separately and then add them to Theta1_grad
%      and Theta2_grad from Part 2.
%



















% =========================================================================

                                % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
