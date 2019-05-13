function y_pred = predictRegress(X, w)
%% PREDICT predicts the Output Label for the given data 

%   Input
%       X  - Input features
%       w  - Regression Parameters
%
%   Output
%       y_pred  - Predicted Output Label for the given data

%% Function starts here

% Output Prediction
y_pred = sigmoid(w'*X')>0.5;

end

