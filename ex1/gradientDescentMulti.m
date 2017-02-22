function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
    J_history(iter) = computeCostMulti(X, y, theta);
    delta = (X' * (X * theta - y) * alpha / m);
    theta = theta - delta;
  end
end
