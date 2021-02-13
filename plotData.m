function plotData (X,y,K)
  
  % indexing your three diagnosis
  cir_idx = find(y == 3); fib_idx = find(y == 2); hep_idx = find(y == 1);
  
  % Plot the data
  cir = scatter3(X(cir_idx,1), X(cir_idx,2), X(cir_idx,3), 30, 'k', 'o');
  hold on;
  fib = scatter3(X(fib_idx,1), X(fib_idx,2), X(fib_idx,3), 30, 'b', 'o');
  hold on;
  hep = scatter3(X(hep_idx,1), X(hep_idx,2), X(hep_idx,3), 30, 'r', 'o');
  
  legend("Cirrhosis","Fibrosis","Hepatitis","location","northeast");
  title("Data plotted in 3D, using PCA for dimensionality reduction");
  
endfunction
