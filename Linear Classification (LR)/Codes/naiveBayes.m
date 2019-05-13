function y_test = naiveBayes(X_train, y_train, X_test)
%% NAIVEBAYES models a Bayesian Classifier to classify the given data.

%   Input
%       X_train  - Train features
%       y_train  - Train labels
%       X_test   - Test features
%
%   Output
%       y_test   - Test labels

%% Function starts here

% Compute Probabilities of two Classes [Spam and Notspam]
P_spam = nnz(y_train)/numel(y_train); % In the labels '1' is Spam & '0' is Notspam
P_notspam = 1 - P_spam;

% Compute Mean & Standrad Deviation of Train features of Spam class
Idx_spam = find(y_train);  % Get Spam Indices
Mu_spam = mean(X_train(Idx_spam,:));
Sigma_spam = std(X_train(Idx_spam,:));

% Compute Mean & Standrad Deviation of Train features of Notspam class
Idx_notspam  =find(~ y_train); % Get Notspam Indices
Mu_notspam = mean(X_train(Idx_notspam,:));
Sigma_notspam = std(X_train(Idx_notspam,:));

% Probabilities of Features of Test data given Class (Spam)
y_test_spam = [];
for i = 1 : size(X_test,1)
    Prob_spam = getProbability(X_test(i, :), Mu_spam, Sigma_spam );
    y_test_spam = [y_test_spam; prod(Prob_spam) * P_spam];
end

% Probabilities of Features of Test data given Class (Notspam)
y_test_notspam = [];
for i = 1 : size(X_test,1)
    Prob_notspam = getProbability(X_test(i,:), Mu_notspam, Sigma_notspam );
    y_test_notspam = [y_test_notspam; prod(Prob_notspam) * P_notspam];
end

% To Check for Higher Probabilities to Asign it to a Class
y_test =  y_test_spam > y_test_notspam;

end

