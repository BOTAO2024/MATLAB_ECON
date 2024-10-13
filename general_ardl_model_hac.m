function [resultsTable_ARDL, resultsTable_HAC] = general_ardl_model_hac(y, X, p_lags, q_lags, hac_lag)
% General ARDL Model Estimation
% !!!!BEFORE YOU USE, CHECK THE FUNCTION 'nwest.m' IS ALREADY IN THE
% SAME PATH.
% Inputs:
% y      - Dependent variable (column vector)
% X      - Independent variable(s) (matrix where each column is a different variable)
% p_lags - Number of lags for the dependent variable (scalar or vector for multiple lags)
% q_lags - Number of lags for each independent variable in X (vector of lags for each column of X)
% hac_lag- Number of lags for Newey-West HAC estimation
%
% BY BOTAO ZHAO
% Last update: 11 OCT 2024
%% Example data

% y = [2.1; 2.5; 2.8; 3.1; 3.6; 4.0; 4.2; 4.8; 5.1; 4.9];  % Dependent variable
% X = [1.2, 0.8; 1.4, 1.0; 1.6, 1.2; 1.8, 1.3; 3.1, 4.1; 2.0, 1.5; 2.2, 1.6; 2.3, 1.7; 2.5, 1.8; 2.2, 1.5];  % Independent variables (multiple columns)

% Set lag lengths for the dependent variable and each independent variable
% p_lags = 2;  % 2 lags for the dependent variable y
% q_lags = [1, 2];  % 1 lag for the first column of X, 2 lags for the second column of X
% hac_lag = 1;

% Call the ARDL model function
% [resultsTable_ARDL, resultsTable_HAC] = general_ardl_model_hac(y, X, p_lags, q_lags, hac_lag);

%%

% Call the ARDL model function

% Outputs:
% beta         - Estimated coefficients
% standardErrors - Standard errors of the coefficients
% tStats       - t-statistics of the estimated coefficients
% residuals    - Residuals of the model
% y_pred       - Predicted values

% beta_HAC          - OLS regression coefficients
% HAC_cov           - Newey-West HAC covariance matrix
% standardErrorsHAC - Newey-West corrected standard errors
% tStats            - t-statistics for each coefficient
% resultsTable      - Table displaying coefficients, SE, and t-stats

% Check if p_lags is scalar or vector
if isscalar(p_lags)
    % If scalar, apply same number of lags for all dependent variables
    p_lags = repmat(p_lags, 1, size(y, 2));
end

% Step 1: Create lagged variables for the dependent variable (y)
laggedY = [];
for i = 1:numel(p_lags)
    laggedY_i = lagmatrix(y, 1:p_lags(i));  % Create lagged values for dependent variable
    laggedY = [laggedY, laggedY_i];  % Combine lagged Y variables
end

% Step 2: Create lagged values for each independent variable (X)
laggedX = [];
for i = 1:size(X, 2)
    lag_q = q_lags(i);  % Get the lag for this independent variable
    for j = 0:lag_q
        laggedX_i = lagmatrix(X(:, i), j);  % Create lag X_{t-j}
        laggedX = [laggedX, laggedX_i];  % Combine all lagged X variables
    end
end

% Step 3: Combine lagged y and lagged X into a single design matrix
% Exclude rows with missing values caused by lagging
combinedMatrix = [laggedY, laggedX];  % Combine lagged y and lagged X with interpret
combinedMatrix = combinedMatrix(max(max(p_lags), max(q_lags))+1:end, :);  % Remove NaN rows caused by lagging
y_adjusted = y(max(max(p_lags), max(q_lags))+1:end);  % Adjust the dependent variable accordingly

combinedMatrix_intercept = [ones(size(y_adjusted)), combinedMatrix];
% Step 4: Estimate the ARDL model using OLS (Ordinary Least Squares)
beta = regress(y_adjusted, combinedMatrix_intercept);  % OLS estimation

% Step 5: Compute residuals and predicted values
y_pred = combinedMatrix_intercept * beta;
residuals = y_adjusted - y_pred;

% Step 6: Compute standard errors of the estimated coefficients
residual_variance = var(residuals);  % Variance of the residuals
covMatrix = residual_variance * inv(combinedMatrix_intercept' * combinedMatrix_intercept);  % Covariance matrix
standardErrors = sqrt(diag(covMatrix));  % Standard errors

% Step 7: Compute t-statistics
tStats = beta ./ standardErrors;

% Step 8: Create a table to display the results
variableNames = ["Intercept"; arrayfun(@(i) ['Variable ' num2str(i)], 1:(numel(beta)-1), 'UniformOutput', false)'];
resultsTable_ARDL = table(beta, tStats, ...
    'VariableNames', {'Coefficient', 'tStatistic'}, ...
    'RowNames', variableNames);

% Display the table
disp('ARDL Model Results:');
disp(resultsTable_ARDL);

%----------------
% NW HAC
%----------------
y = y_adjusted;
X = combinedMatrix_intercept;
hac_lag = hac_lag;

% Call Function to compute OLS regression coefficients and Newey-West HAC corrected covariance matrix
results_HAC = nwest(y, X, hac_lag);

rsqr_display = [results_HAC.rsqr, NaN(1,1)];
dw_display = [results_HAC.dw, NaN(1,1)];
% Show the Newey-West HAC Table
results_display = [results_HAC.beta, results_HAC.tstat; rsqr_display; dw_display];

% Define row and column names
rowNames_all = ["Intercept"; arrayfun(@(i) ['Variable ' num2str(i)], 1:(numel(beta)-1), 'UniformOutput', false)'; 'R-squre'; 'DW test'];

columnNames_all = {'Coefficient', 'tStat'};

% Create the table
resultsTable_HAC = array2table(results_display, 'RowNames', rowNames_all, 'VariableNames', columnNames_all);
disp('Newey-West HAC Results:')
disp (resultsTable_HAC);

% Display the significance level criteria
fprintf('Significance levels based on t-statistics:\n');
fprintf('|t| > 1.645 for 10%% significance\n');
fprintf('|t| > 1.96 for 5%% significance\n');
fprintf('|t| > 2.576 for 1%% significance\n');

end

