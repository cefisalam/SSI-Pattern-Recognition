function w = regressionParameter(X, y, Lambda)
%% REGRESSIONPARAMETER computes Optimal values for 'w' using Regularized Cost function

%   Input
%       X      - Input features
%       y      - Output Label
%       Lambda - Regularization Parameter
%
%   Output
%       w      - Regression Parameters

%% Function starts here

% Initialize Regression Parameters
w = zeros(size(X,2),1);

% Minimize the Cost Function using Gradient Descent (Here we use Built-in Optimizer)
options = optimoptions('fminunc','GradObj','on','MaxIter',400);

[w,~] = fminunc(@(w) costFunction(X, y, w, Lambda), w, options);

end

