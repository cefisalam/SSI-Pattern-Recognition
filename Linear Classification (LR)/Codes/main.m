close all;
clear; clc;
%% Load the data

% Train data
Xtrain = load('./SpamData/spamTrain.txt');
ytrain = load('./SpamData/spamTrainLabels.txt');

% Test data
Xtest = load('./SpamData/spamTest.txt');
ytest = load('./SpamData/spamTestLabels.txt');

%% Part - I: Max and Mean of the Average length of Uninterrupted Sequences of Capital letters

[Max, Mean] = getMaxMean(Xtrain(:,55));

disp(['The Max of the Average length of Uninterrupted Sequences of Capital letters in the Training set is ',num2str(Max)]);
disp(['The Mean of the Average length of Uninterrupted Sequences of Capital letters in the Training set is ',num2str(Mean)]);

%% Part - II: Max and Mean of the lengths of Longest Uninterrupted Sequences of Capital letters

[Max1, Mean1] = getMaxMean(Xtrain(:,56));

disp(['The Max of the lengths of Longest Uninterrupted Sequences of Capital letters in the Training set is ',num2str(Max1)]);
disp(['The Mean of the lengths of Longest Uninterrupted Sequences of Capital letters in the Training set is ',num2str(Mean1)]);

%% Part - III: Preprocessing of Input Features

% Standardize the Columns to have Zero Mean and Unit Variance
X_strd = preProcess([Xtrain;Xtest], 'Standardize');
Xtrain_strd = X_strd(1:size(Xtrain,1),:); % Train data
Xtest_strd = X_strd(size(Xtrain,1)+1:end, :); % Test data

% Transform the Features Using log(xij + 0.1)
Xtrain_log = preProcess(Xtrain, 'Log'); % Train data
Xtest_log = preProcess(Xtest, 'Log'); % Test data

% Binarize the features using I(xij > 0)
Xtrain_bin = preProcess(Xtrain, 'Binarize'); % Train data
Xtest_bin = preProcess(Xtest, 'Binarize'); % Test data

%% Part - IV: Logistic Regression with Regularization

%% Standardized Input Features

% Add the bias term (column of ones before data)
Xtrain_strd_LR = [ones(size(Xtrain_strd,1),1) Xtrain_strd];
Xtest_strd_LR = [ones(size(Xtest_strd,1),1) Xtest_strd];

% Compute Lambda using Cross Validation
lambda_strd = getLambda(Xtrain_strd_LR, ytrain);

% Compute the Regression Parameter (w)
w_strd = regressionParameter(Xtrain_strd_LR, ytrain, lambda_strd);

% Calculate the Prediction Error for Train data
ytrain_strd_LR = predictRegress(Xtrain_strd_LR, w_strd);
Etrain_strd_LR = errorPredict(ytrain_strd_LR, ytrain);
disp(['Prediction Error for Standardized Train data: ',num2str(Etrain_strd_LR)]);

% Calculate the Prediction Error for Test data
ytest_strd_LR = predictRegress(Xtest_strd_LR, w_strd);
Etest_strd_LR = errorPredict(ytest_strd_LR, ytest);
disp(['Prediction Error for Standardized Test data: ',num2str(Etest_strd_LR)]);

%% Log Transformed Input Features

% Add the bias term
Xtrain_log_LR = [ones(size(Xtrain_log,1),1) Xtrain_log];
Xtest_log_LR = [ones(size(Xtest_log,1),1) Xtest_log];

% Compute Lambda using Cross Validation
lambda_log  = getLambda( Xtrain_log_LR, ytrain );

% Compute the Regression Parameter (w)
w_log = regressionParameter(Xtrain_log_LR, ytrain, lambda_log);

% Calculate the Prediction Error for Train data
ytrain_log_LR = predictRegress(Xtrain_log_LR, w_log);
Etrain_log_LR = errorPredict(ytrain_log_LR, ytrain);
disp(['Prediction Error for Log Transformed Train data: ',num2str(Etrain_log_LR)]);

% Calculate the Prediction Error for Test data
ytest_log_LR = predictRegress(Xtest_log_LR, w_log);
Etest_log_LR = errorPredict(ytest_log_LR, ytest);
disp(['Prediction Error for Log Transformed Test data: ',num2str(Etest_log_LR)]);

%% Binarized Input Features

% Add the bias term
Xtrain_bin_LR = [ones(size(Xtrain_bin,1),1) Xtrain_bin];
Xtest_bin_LR = [ones(size(Xtest_bin,1),1) Xtest_bin];

% Compute Lambda using Cross Validation
lambda_bin  = getLambda( Xtrain_bin_LR, ytrain);

% Compute the Regression Parameter (w)
w_bin = regressionParameter( Xtrain_bin_LR, ytrain, lambda_bin);

% Calculate the Prediction Error for Train data
ytrain_bin_LR = predictRegress(Xtrain_bin_LR, w_bin);
Etrain_bin_LR = errorPredict(ytrain_bin_LR, ytrain);
disp(['Prediction Error for Binarized Train data: ',num2str(Etrain_bin_LR)]);

% Calculate the Prediction Error for Test data
ytest_bin_LR = predictRegress(Xtest_bin_LR, w_bin);
Etest_bin_LR = errorPredict(ytest_bin_LR, ytest);
disp(['Prediction Error for Binarized Test data: ',num2str(Etest_bin_LR)]);

%% Part - V: Naive Bayes Classifier

%% Standardized Input Features

% Calculate the Prediction Error for Test data
ytrain_strd_NB = naiveBayes(Xtrain_strd, ytrain, Xtrain_strd);
Etrain_strd_NB = errorPredict(ytrain_strd_NB, ytrain);
disp(['Prediction Error for Standardized Train data (Naive Bayes): ',num2str(Etrain_strd_NB)]);

% Calculate the Prediction Error for Test data
ytest_strd_NB = naiveBayes(Xtrain_strd, ytrain, Xtest_strd);
Etest_strd_NB = errorPredict(ytest_strd_NB, ytest);
disp(['Prediction Error for Standardized Test data (Naive Bayes): ',num2str(Etest_strd_NB)]);

%% Log Transformed Input Features

% Calculate the Prediction Error for Test data
ytrain_log_NB = naiveBayes(Xtrain_log, ytrain, Xtrain_log);
Etrain_log_NB = errorPredict(ytrain_log_NB, ytrain);
disp(['Prediction Error for Log Transformed Train data (Naive Bayes): ',num2str(Etrain_log_NB)]);

% Calculate the Prediction Error for Test data
ytest_log_NB = naiveBayes(Xtrain_log, ytrain, Xtest_log);
Etest_log_NB = errorPredict(ytest_log_NB, ytest);
disp(['Prediction Error for Log Transformed Test data (Naive Bayes): ',num2str(Etest_log_NB)]);

%% Binarized Input Features

% Calculate the Prediction Error for Test data
ytrain_bin_NB = naiveBayes(Xtrain_bin, ytrain, Xtrain_bin);
Etrain_bin_NB = errorPredict(ytrain_bin_NB, ytrain);
disp(['Prediction Error for Binarized Train data (Naive Bayes): ',num2str(Etrain_bin_NB)]);

% Calculate the Prediction Error for Test data
ytest_bin_NB = naiveBayes(Xtrain_bin, ytrain, Xtest_bin);
Etest_bin_NB = errorPredict(ytest_bin_NB, ytest);
disp(['Prediction Error for Binarized Test data (Naive Bayes): ',num2str(Etest_bin_NB)]);

%% Uncomment the following to use built-in function for Naive Bayes Classifier

% %% Standardized Input Features
% 
% % Train the Model
% Md1 = fitcnb(Xtrain_strd, ytrain);
% 
% % Calculate the Prediction Error for Train data
% ytrain_strd_NB = predict(Md1, Xtrain_strd);
% Etrain_strd_NB = errorPredict(ytrain_strd_NB, ytrain);
% disp(['Prediction Error for Standardized Train data: ',num2str(Etrain_strd_NB)]);
% 
% % Calculate the Prediction Error for Test data
% ytest_strd = predict(Md1, Xtest_strd);
% Etest_strd_NB = errorPredict(ytest_strd, ytest);
% disp(['Prediction Error for Standardized Test data: ',num2str(Etest_strd_NB)]);
% 
% %% Log Transformed Input Features
% 
% % Train the Model
% Md2 = fitcnb(Xtrain_log, ytrain);
% 
% % Calculate the Prediction Error for Train data
% ytrain_log_NB = predict(Md2, Xtrain_log);
% Etrain_log_NB = errorPredict(ytrain_log_NB, ytrain);
% disp(['Prediction Error for Log Transformed Train data: ',num2str(Etrain_log_NB)]);
% 
% % Calculate the Prediction Error for Test data
% ytest_log = predict(Md2, Xtest_log);
% Etest_log_NB = errorPredict(ytest_log, ytest);
% disp(['Prediction Error for Log Transformed Test data: ',num2str(Etest_log_NB)]);