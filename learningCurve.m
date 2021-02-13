function [error_train, error_val] = ...
    learningCurve(X_train, y_train, Xval, yval, lambda)
%   Generate training and validation error for plotting of learning curve
%   It will return vectors error_train and error_val. error_train/val(i) contains
%   training error for i examples

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);


  for i = 1:m,
    theta = trainLinearReg(X(1:i, :),y(1:i),lambda); 
    % returning theta for subset of training
    lambda_lc = 0;
    % set lambda to 0
    
    % Note that you limit the training set for your training error not your validation
    % Your validation should always include the entire validation dataset, likewise for your
    % test set if you are calculating errors
    % Your training error is essentially the exact same as your cost function, but without regularization
    
    error_train(i) = linearRegCostFunction(X(1:i, :),y(1:i),theta,lambda_lc);
    error_val(i) = linearRegCostFunction(Xval,yval,theta,lambda_lc);
  endfor






% -------------------------------------------------------------

% =========================================================================

end
