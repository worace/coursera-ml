%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);



%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction()');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);



%% ================ Part 3: Predict for One-Vs-All ================

fprintf('Predicting based on training thetas...\n')
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% Test Cases (https://www.coursera.org/learn/machine-learning/discussions/all/threads/5g8LaZTCEeW0dw6k4EUmPw) %%
%%%%%%%%%%%%%%%%%%%
% 1. lrCostFunction
%%%%%%%%%%%%%%%%%%%
theta = [-2; -1; 1; 2];
X = [ones(5,1) reshape(1:15,5,3)/10];
y = [1;0;1;0;1] >= 0.5;       % creates a logical array
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda)

% Expected:
## J =  2.5348
## grad =
##    0.14656
##   -0.54856
##    0.72472
##    1.39800


%%%%%%%%%%%%%%%%%%%
% 2. oneVsAll
%%%%%%%%%%%%%%%%%%%
X = [magic(3) ; sin(1:3); cos(1:3)];
y = [1; 2; 2; 1; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda)
%expected:
## all_theta =
##   -0.559478   0.619220  -0.550361  -0.093502
##   -5.472920  -0.471565   1.261046   0.634767
##    0.068368  -0.375582  -1.652262  -1.410138

%%%%%%%%%%%%%%%%%%%
% 3. predictOneVsAll
%%%%%%%%%%%%%%%%%%%

all_theta = [1 -6 3; -2 4 -3];
X = [1 7; 4 5; 7 8; 1 4];
predictOneVsAll(all_theta, X)
%output:
## ans =
##    1
##    2
##    2
##    1


%%%%%%%%%%%%%%%%%%%
% 4. predict
%%%%%%%%%%%%%%%%%%%

Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X)
% you should see this result
## p =
##   4
##   1
##   1
##   4
##   4
##   4
##   4
##   2
