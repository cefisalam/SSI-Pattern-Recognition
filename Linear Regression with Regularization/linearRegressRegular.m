function [Erms, Norm_W, W] = linearRegressRegular(X, y, M, Lambda)
%% LINEARREGRESS computes Rugularized Regression parameters and plots best fit for the given data

%   Input
%       X       - Input Feature data
%       y       - Output data
%       M       - Degree of Polynomial to fit the data
%       Lambda  - Regularization parameter
%
%   Output
%       Erms    - Root-Mean-Square Error
%       Norm_W  - Norm of the Regression Parameters
%       W       - Regression Parameters

%% Function starts here

% Initialize the basis function (phi(X))
phi = ones(size(X,1),1);

for i = 1:M
    phi = [phi X.^i]; % Feature data for different Degree of Polynomial
end

% Compute the Regularized Regression parameters
W = pinv((Lambda*eye(M+1)) + (phi'*phi)) * phi' * y; % Using the Formula W = ((Lambda*I)+(phi(X)'*phi(X)))^-1*phi(X)'*y

% Compute the Output Estimate using the Regularized Regression parameter
y_new = W' * phi';

%% To compute RMS Error

% Empirical error
temp = (y_new - y').^2;
E_emp = 0.5*sum(temp);

% RMS Error
Erms = sqrt(2*E_emp/length(X));

% To compute Norm of the Regularized Regression parameter
Norm_W = norm(W)^2;

%% Plot the Best fit

figure,
scatter(X,y,'b+');
hold on;
plot(X,y_new,'r--');
xlabel('Feature data (X)');
ylabel('Output data (y)');
title('Linear Regression with Regularization(Best Fit)');

end