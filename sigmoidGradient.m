function g = sigmoidGradient(z)
% Compute gradient of sigmoid(z), i.e the derivative of the sigmoid activation function
% Used to subsequently calculate error terms

%   This should work regardless if z is a matrix, vector or scalar.
%   In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

g = 1 ./(1 + exp(-z));

g = g .* (1-g);

% =============================================================




end
