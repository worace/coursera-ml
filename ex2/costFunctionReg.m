function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



m = length(y);

h = sigmoid(X*theta);
reg_cost =  sum(theta(2:end).^2)/m/2 * lambda;
J = (-y' * log(h) - (1-y')*log(1-h))/m + reg_cost;
% don't regularize first theta term (constant term)
reg = [0; (lambda/m) * theta(2:end)];
grad = (X' * (h - y))/m + reg;

% =============================================================

end
