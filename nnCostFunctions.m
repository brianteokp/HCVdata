function [J grad] = nnCostFunctions(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% This function returns the cost function and the gradient to be fed into fmincg
% for optimization of the NN. This works for a classification neural network 
% with 1 hidden layer
                             
% reshaping unrolled parameters                                  
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
m = size(X,1); 

% Returning these variables 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));  


% Part 1: Feedforward Neural Network and returns the cost associated with 
% the weights
% Calculates the hypothesis

X = [ones(m,1) X]; % Adding bias terms to input layer              
z2 = Theta1 * X'; 
a2 = sigmoid(z2); % Values of first activation unit

a2 = [ones(m,1) a2']; % Adding bias terms
z3 = Theta2 * a2'; 
h = sigmoid(z3); % Values of hypothesis

% Repurposing vector y such that it each y value is changed to a k-dimensional vector
% where k = number of labels
recode_y = zeros(num_labels, m); 

  for i = 1:m,
    
    recode_y(y(i),i) = 1; 
    
  endfor

% Calculating cost function
J = (1/m) *  sum(sum((-recode_y .* log(h)) - ((1 .- recode_y) .* log(1-h))));

% Removing bias term for use in regularization calculations
temp1 = Theta1;  
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;

% Adding regularization term for cost function
reg =(lambda/(2*m)) * (sum(sum(temp1.^2)) + sum(sum(temp2 .^ 2)));

J = J + reg;


% Part 2: Back propogation to compute the gradient Theta1_grad, Theta2_grad
% which are the partial derivatives of the cost function for layer 2 and layer 3.



% Calculating activations for every layer for every training example
 for t = 1:m,
  
  % Step 1: activations of every layer per training example
  a1 = X(t,:); 
  a1 = a1';
  z2 = Theta1 * a1 ; 
  a2 = sigmoid(z2);
  a2 = [1;a2]; % Adding bias term
  
  z3 = Theta2 * a2 ; 
  a3 = sigmoid(z3);
  
  % Step 2: delta of output layer
  yk = recode_y(:,t); 
  d3 = a3 - yk; 
  
  % Step 3: Back propogation
  z2 = [1;z2]; % Adding bias term
  
  d2 = (Theta2' * d3) .* sigmoidGradient(z2); 
  d2 = d2(2:end); % removing the error for bias term
  
  Theta2_grad = Theta2_grad + d3 * a2'; 
  Theta1_grad = Theta1_grad + d2 * a1';
  
    
endfor



Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

% Adding regularization term for gradient
reg_1 = (lambda/m) * temp1;
reg_2 = (lambda/m) * temp2;

Theta2_grad = Theta2_grad + reg_2;
Theta1_grad = Theta1_grad + reg_1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction

