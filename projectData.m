function Z = projectData (U, X, K)
% This function aims to compute the m x k matrix Z, which is the projection
% of X onto the top K eigenvectors (U)

m = size(X,1);

U_reduce = U(:,1:K); %subsetting out top K eigenvectors, n x K 
Z = zeros(m, K); % m x K

Z = X * U_reduce; % m x K


endfunction
