function E_pred = errorPredict(y_pred, y)
%% ERRORPREDICT calculates the Prediction Error

%   Input
%       y      - True Output Label for the given data
%       y_pred - Predicted Output Label for the given data
%
%   Output
%       E - Prediction Error (%)

%% Function starts here

% Initialize Error 
E = 0;

for i = 1:length(y_pred)
    
    if y_pred(i) ~= y(i)
        E = E + 1;
    end
    
end

% Error Percentage
E_pred = E/length(y_pred)*100;

end

