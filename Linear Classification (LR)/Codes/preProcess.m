function Data_Out = preProcess(Data, Method)
%% PREPROCESS performs the required preprocessing for the given Dataset

%   Input
%       Data   - Input data
%       Method - Preprocessing Method {'Standardize', 'Log', 'Binarize'}
%
%   Output
%       Data_Out - Processed Data
%       Mu       - Mean of the given data
%       Sigma    - Standard Deviation of the given data

%% Function starts here

switch Method
    case 'Standardize'
        % Standardize the columns so they all have mean 0 and unit variance
        
        % Compute Mean
        Mu = mean(Data);
        
        % Compute Standard Deviation
        Sigma = std(Data);
        
        % Convert Mu and Sigma to Matrix form
        Mu_Matrix = repmat(Mu, size(Data,1),1);
        Sigma_Matrix = repmat(Sigma, size(Data,1),1);
        
        % Normalize the data
        Data_Out = (Data - Mu_Matrix) ./ Sigma_Matrix;
        
    case 'Log'
        % Transform the features using log(Xij + 0.1) [Add 0.1 to each feature to avoid taking log of zero]
        
        Data_Out = log(Data+0.1);
        
    case 'Binarize'
        % Binarize the features using I(xij > 0)
        
        Data_Out = Data > 0;
end

end

