% This part of the project uses PCA to visualize our dataset in 3D
clc;
clear;

data = dlmread("hcvdat0.csv");

y = data(2:end,1);
X = data(2:end, 2:end);


% Normalizing X
% mu and sigma gives the mean and std of each feature
[X_norm, mu, sigma] = featureNormalize(X);


% Running PCA to reduce down to 3-dimensions for data visualization
[U,S] = pca(X_norm);

fprintf('Top 3 eigenvectors for data visualization \n');
fprintf('\n U(:,1) = \n')
fprintf('%f \n', U(:,1));

fprintf('\n U(:,2) = \n')
fprintf('%f \n', U(:,2));

fprintf('\n U(:,3) = \n')
fprintf('%f \n', U(:,3));

fprintf('\n Program paused');
pause;

clc;

% Projecting the data on 3-dimensions
K = 3;
Z = projectData(U, X_norm, K);

plotData(Z,y,K);
hold on;
% Visually, doesn't look like the data will be easily separable ... 
% But lets try training a NN anyway!








  