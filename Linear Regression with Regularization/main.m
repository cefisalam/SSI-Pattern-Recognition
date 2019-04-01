close all; 
clear; clc;

%% Load the data

% Train data
Train = load('./hw1training.txt');
X_train = Train(:,1);
y_train = Train(:,2);

% Test data
Test = load('./hw1test.txt');
X_test = Test(:,1);
y_test = Test(:,2); 

%% Part - I: Linear Regression for a Polynomial Function

M = [0, 2, 3, 5, 8, 10]; % Degree of Polynomial to fit the data

for i = 1:length(M)
    
    % Compute the Parameters for different values of M
    [Erms, W] = linearRegression(X_train, y_train, M(i));
end

%% Part - II: Select a "good" Model

M = 0:15; % Degree of Polynomial to fit the data

Erms_train = [];
Erms_test = [];

for i = 1:length(M)
    
    % Compute the Parameters for Training data
    [Erms1, W1] = linearRegression(X_train, y_train, M(i));
    Erms_train = [Erms_train Erms1];
    
    % Compute the Parameters for Test data
    [Erms2, W2] = linearRegression(X_test, y_test, M(i));
    Erms_test = [Erms_test Erms2];
end

% Plot RMS Error for Training and Test data
figure,
M = 0:1:15;
plot(M,Erms_train,'r-',M,Erms_test,'b--');
xlabel('M');
ylabel('RMS Error');
legend('Train data','Test data');

%% Part - III: Linear Regression with Regularization

M = 10;

Lambda = 0.001;

Erms_train_reg = [];
Erms_test_reg = [];
Norm_W_train = [];
Norm_W_test = [];

for j = linspace(0,Lambda,10)
    
    % Compute the Parameters for Training data
    [Erms1, Norm_W1, ~] = linearRegressRegular(X_train, y_train, M, j);
    Erms_train_reg = [Erms_train_reg Erms1];
    Norm_W_train = [Norm_W_train Norm_W1];

    % Compute the Parameters for Training data
    [Erms2, Norm_W2, ~] = linearRegressRegular(X_test, y_test, M, j);
    Erms_test_reg = [Erms_test_reg Erms2];
    Norm_W_test = [Norm_W_test Norm_W2];
end

% Plot RMS Error for Training and Test data
figure,
M = linspace(0,Lambda,10);
plot(M,Erms_train_reg,'r-',M,Erms_test_reg,'b--');
xlabel('Lambda');
ylabel('RMS Error');
legend('Train data','Test data');

% Plot Norm for Training and Test data
figure,
M = linspace(0,Lambda,10);
plot(M,Norm_W_train,'r-',M,Norm_W_test,'b--');
xlabel('Lambda');
ylabel('Norm');
legend('Train data','Test data');
