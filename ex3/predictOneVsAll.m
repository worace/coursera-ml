function p = predictOneVsAll(all_theta, X)
  # X: 5000 x 400 examples matrix
  # all_theta
  # 10 x 401 parameter weight matrix
  # (10 classifiers x 400 features + 1 bias feature)
  # size([ones(size(X,1),1) X] * all_theta')
  # 5000     10
  # Output:
  # 5000 x 1 vector of classifications for each
  # input
  # X * all_theta' -- gives matrix of one prediction per class for each example
  # then, choose the Max output for each row

  per_class_predictions = [ones(size(X,1),1) X] * all_theta';
  per_class_predictions = sigmoid(per_class_predictions);
  [confidences, classes] = max(per_class_predictions, [], 2);
  fprintf('size of p: %f\n', size(classes));
  p = classes;

end
