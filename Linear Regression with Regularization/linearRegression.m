function [Erms, W] = linearRegression(X, y, M)
%% LINEARREGRESSION computes Regression parameters and plots best fit for the given data

%   Input
%       X  - Input Feature data
%       y  - Output data
%       M  - Degree of Polynomial to fit the data
%
%   Output
%       Erms - Root-Mean-Square Error
%       W    - Regression Parameters

%% Function starts here

% Initialize the basis function (phi(X))
phi = ones(size(X,1),1);

for i = 1:M   
    phi = [phi X.^i]; % Feature data for different Degree of Polynomial
end

% Compute the Regression parameters
W = pinv(phi'*phi) * phi' * y; % Using the Formula W = (phi(X)'*phi(X))^-1*phi(X)'*y

% Compute the Output Estimate using the Regression parameters
y_new = W' * phi';

%% To compute RMS Error

% Empirical error
temp = (y_new - y').^2;
E_emp = 0.5*sum(temp);

% RMS Error
Erms = sqrt(2*E_emp/length(X));

%% Plot the Best fit

figure,
scatter(X,y,'r*');
hold on;
plot(X,y_new,'b-');
xlabel('Feature data (X)');
ylabel('Output data (y)');
title('Linear Regression (Best Fit)');

end

