function p = predict(Theta1, Theta2, X)
  % Given trained weights Theta1 and Theta2, output predicted label of X.

  m = size(X, 1);
  num_labels = size(Theta2, 1);
  
  p = zeros(size(X, 1), 1); % m-dim vector
  
  
  X1 = [ones(m,1) X]; % Adding bias terms
  z2 = X1 * Theta1';
  a2 = sigmoid(z2);

  a2 = [ones(size(a2,1),1) a2]; % Adding bias terms
  z3 = Theta2 * a2';
  a3 = sigmoid(z3);
  a3 = a3';

  [a b] = max(a3, [], 2); 
  % Take the index of the highest value in the output layer
  % This maximum value = the diagnosis the NN is most confident in
  p = b;

endfunction
