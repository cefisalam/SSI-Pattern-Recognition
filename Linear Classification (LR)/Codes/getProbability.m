function Prob = getProbability(X, Mu, Sigma)
%% GETPROBABILITY computes Conditional Probability of the given Input feature.

%   Input
%       X      - Input features
%       Mu     - Mean
%       Sigma  - Standard Deviation
%
%   Output
%       Prob   - Conditional Probability P(X/Class)

%% Function starts here

% Compute the Probability of the Normal Distribution
Prob = exp(-(X - Mu).^2 ./ (2 .* Sigma.^2)) ./ (sqrt(2 * pi) .* Sigma);

end

