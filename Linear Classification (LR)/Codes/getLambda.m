function Lambda = getLambda(X,y)
%% GETBESTLAMBDA computes Optimal value for Regularization Parameter (Lambda) using Cross Validation

%   Input
%       X      - Input features
%       y      - Output Label
%
%   Output
%       Lambda - Regularization Parameter

%% Function starts here

% Divide the given features into Training (80%) and Validation (20%) set
C = ceil(0.8*size(X,1));

% Trainig Set
X_train = X(1:C,:);
y_train = y(1:C,:);

% Validation Set
X_valid = X(C:end,:);
y_valid = y(C:end,:);

Idx = 1;

lambda = linspace(0,0.001,11); % Different Initial Values for Lambda between [0,0.1]

for i = lambda 
        
    % Compute Regression Parameters Using Optimisation
    w = regressionParameter(X_train, y_train, i);
    
    % Predict the Label for Training set using Estimated Regression Parameters
    y_pred_train = predictRegress(X_train, w);
        
    % Predict the Label for Validation set
    y_pred_valid = predictRegress(X_valid, w);
    
    % Compute Training and Validation Errors
    Error_train(Idx) = errorPredict(y_pred_train, y_train);
    Error_valid(Idx) = errorPredict(y_pred_valid, y_valid);
    Idx = Idx + 1;    
    
end

% Choose the Lambda with minimum Validation error
[Val,idx] = min(Error_valid);
Lambda = lambda(idx);

% Plot
figure,
plot(lambda,Error_train,'b', lambda,Error_valid,'g');
hold on,
stem(Lambda,Val,'r--', 'Marker', '*');
xlabel('Lambda');ylabel('Error (in %)');title('Training & Validation Errors');
legend('Training Error','Validation Error','Best Lambda');

end
