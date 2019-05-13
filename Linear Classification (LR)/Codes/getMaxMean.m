function [Max_X, Mean_X] = getMaxMean(X)
%% GETMAXMEAN computes the "Max and Mean" of the given Data

%   Input
%       X  - Input data
%
%   Output
%       Max_X  - Max of the given data
%       Mean_X - Mean of the given data

%% Function starts here

% Compute Max
Max_X = max(X);

% Compute Mean
Mean_X = mean(X);

end

