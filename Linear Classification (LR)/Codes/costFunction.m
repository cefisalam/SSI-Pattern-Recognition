function [E, Grad] = costFunction(X, y, w, Lambda)
%% COSTFUNCTION computes Regularized Cost and Gradient functions for given data

%   Input
%       X      - Input features
%       y      - Output Label
%       w      - Regression Parameters
%       Lambda - Regularization Parameter
%
%   Output
%       E    - Computed Cost
%       Grad - Computed Gradient

%% Function starts here

% Initialize Cost and Gradient
[r,c] = size(X);
E = 0;
Grad = zeros(r,c);

for i = 1:r % Compute for all samples
    
    phi_X = sigmoid(w'*X(i,:)');
    
    E = E + (y(i)*log(phi_X) + (1-y(i))*log(1-phi_X)); % Cost for Each Sample
    
    Grad(i,:) = (phi_X-y(i))*X(i,:) + Lambda.*[0, w(2:end)']; % Gradient for Each Sample
end

% Cost for the entire data (with Regularization)
E = (-1*E/r) + (Lambda/2*sum(w(2:end).^2));

% Gradient for the entire data (with Regularization)
Grad = sum(Grad)/r;

end

