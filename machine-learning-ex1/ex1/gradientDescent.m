function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

    J_history(iter) = computeCost(X, y, theta);

    % theta [0 0]
    % X
    % 1 6.11
    % 1 5.52
    % 1 8.51

    % X'
    % 1    1    1
    % 6.11 5.52 8.51

    % 1. get cost (sum X .* theta - y)
    % 2. divide by m
    % 3. multiply by X(i)

    ## Vectorized:
    delta = X' * (X*theta - y) * alpha/m;
    theta = theta - delta;
    ## Iterative:
    ## theta_new = zeros(length(theta),1);
    ## for i = 1:length(theta)
    ##   theta_new(i) = theta(i) - alpha/m * sum(((X * theta) - y) .* X(:,[i]));
    ## end
    ## theta = theta_new;
  end

end
