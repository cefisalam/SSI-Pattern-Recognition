function Sig_X = sigmoid(X)
%% SIGMOID computes the Sigmoid Function for the given data

%   Input
%       X  - Input features
%
%   Output
%       Sig_X  - Sigmoid Function for the given data

%% Function starts here

% Compute Sigmoid Function
Sig_X = 1./(1+exp(-X));

end

