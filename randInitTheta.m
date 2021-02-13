function theta = randInitTheta (L_in, L_out)
  % randomly initalizes weights for a particular layer based on dimensions given
  % by L_in (input layer) and L_out (output layer)
  % For e.g for hidden layer 1, L_in = size of input feature layer, 
  % L_out = size of hidden layer 1
  
  theta = zeros(L_out, L_in + 1); % +1 because of bias term!
  
  init_E = sqrt(6)/ (sqrt(L_in + L_out));
  
  theta = rand(L_out,1+L_in) * 2 * init_E - init_E;

  
endfunction
