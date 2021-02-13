function [nn_params] = trainNN (X, y, lambda,init_nn_params)

nn_params = zeros(size(init_nn_params,1),1);

options = optimset('MaxIter', 50);

% costFunction seeks to find optimal weights, w
costFunction = @(w) nnCostFunctions(w, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Running fmincg to find the optimal weights, returns optimal weights in nn_params
[nn_params, cost] = fmincg(costFunction, init_nn_params, options);

endfunction
