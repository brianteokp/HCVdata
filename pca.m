function [U, S] = pca(X)
%   Run PCA on X (should be normalized beforehand)
%   U returns eigenvector, S returns eigenvalues


[m, n] = size(X);

U = zeros(n);
S = zeros(n);

sigma = (1/m) * (X' * X); %covariance matrix, n x n dimension


[U, S, V] = svd(sigma);



% =========================================================================

end
