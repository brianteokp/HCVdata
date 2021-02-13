% This part of the project trains a simple NN 
% with 1 hidden layer (10 activation units)

clc;
clear;


data = dlmread("hcvdat0.csv");
data = data(2:end,:);
data_shuffle = data(randperm(size(data,1)),:);

% Shuffling ordered data
y = data_shuffle(2:end,1);
X = data_shuffle(2:end, 2:end);

% Parameters for the project
[m,n] = size(X);
input_layer_size = 10;
hidden_layer_size = 10;
num_labels = 3;

% Normalizing X
% mu and sigma gives the mean and std of each feature
[X_norm, mu, sigma] = featureNormalize(X);

% Splitting dataset into training, validation and testing set
% 60-20-20 split
X_train = X(1:33,:); y_train = y(1:33,:);
X_val = X(34:44,:); y_val = y(34:44,:);
X_test = X(45:55,:); y_test = y(45:55,:);



%%%%%%%%%%%%%%%%%%%%% PART 1: Random Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin by randomly initializing some parameters for forward propogation

initial_Theta1 = randInitTheta(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitTheta(hidden_layer_size,num_labels);

% Unrolling parameters
init_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%%%%%%%%%%%%%%%%%%%%% PART 2: Code for NN for J and grad %%%%%%%%%%%%%%%%%%%%%%%
% Code is in nnCostFunctions.m

% Checking gradients first to ensure the neural network is working fine.
% Taken from Prof Andrew Ng, includes the package:
% checkNNGradients.m, computeNumericalGradient.m, debugInitializeWeights.m

fprintf('\nPerform Gradient Checking \n')

lambda = 0;
checkNNGradients(lambda);

fprintf('\nProgram paused\n')

pause;
clc;

% computeNumericalGradient performs the gradient checking function, working with
% e = 1e-4

%%%%%%%%%%%%%%%%%%%%% PART 3: Running fmincg with J and grad %%%%%%%%%%%%%%%%%%%

fprintf('\nTraining Neural Network... \n')

% Set number of iterations here
options = optimset('MaxIter', 1000);

% costFunction seeks to find optimal weights, w
costFunction = @(w) nnCostFunctions(w, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Running fmincg to find the optimal weights, returns optimal weights in nn_params
[nn_params, cost] = fmincg(costFunction, init_nn_params, options);

% Reshaping to obtain optimal weights
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nProgram paused\n')
pause;
clc;

%%%%%%%%%%%%%%%%%%%%% PART 4: Determining Training Accuracy %%%%%%%%%%%%%%%%%%%%

pred = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

fprintf('\nProgram ended.\n');
pause;


