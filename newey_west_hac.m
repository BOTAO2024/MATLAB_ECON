function [beta, HAC_cov, standardErrorsHAC, tStats, resultsTable] = newey_west_hac(X, y, lag)
    % Function to compute OLS regression coefficients and Newey-West HAC corrected covariance matrix
    %
    % Inputs:
    % X   - Independent variables matrix (including a column for the intercept if needed)
    % y   - Dependent variable vector
    % lag - Lag length for Newey-West HAC estimator
    %
    % Outputs:
    % beta              - OLS regression coefficients
    % HAC_cov           - Newey-West HAC covariance matrix
    % standardErrorsHAC - Newey-West corrected standard errors
    % tStats            - t-statistics for each coefficient
    % resultsTable      - Table displaying coefficients, SE, and t-stats
    %
    % BY BOTAO ZHAO
    % Last update: 11 OCT 2024
    
    % Step 1: Fit an OLS regression model
    X = [ones(size(y)), X];
    beta = regress(y, X);

    % Step 2: Compute the residuals
    residuals = y - X * beta;

    % Step 3: Compute the Newey-West HAC covariance matrix
    [n, k] = size(X);  % n: sample size, k: number of predictors

    % Initial covariance matrix for heteroscedasticity (S0)
    S0 = (X' * diag(residuals.^2) * X) / n;

    % Initialize the HAC covariance matrix with S0
    HAC_cov = S0;

    % Compute the weighted lag contributions
    for l = 1:lag
        weight = 1 - (l / (lag + 1));  % Newey-West weighting scheme

        % Compute the lagged residual contributions
        Gamma_l = (X(l+1:n,:)' * diag(residuals(1:n-l)) * X(1:n-l,:)) / n;
        % Add the weighted lagged terms
        HAC_cov = HAC_cov + weight * (Gamma_l + Gamma_l');
    end

    % Step 4: Newey-West corrected standard errors
    standardErrorsHAC = sqrt(diag(HAC_cov));

    % Step 5: Compute t-statistics (beta divided by standard errors)
    tStats = beta ./ standardErrorsHAC;

    % Step 6: Create a table of results
    VariableNames = ["Intercept"; arrayfun(@(i) ['Variable ' num2str(i)], 1:(numel(beta)-1), 'UniformOutput', false)'];
    resultsTable = table(beta, standardErrorsHAC, tStats, 'VariableNames', {'Coefficient', 'StandardError', 'tStatistic'}, 'RowNames', VariableNames);

    % Display the table
    disp('Newey-West HAC Results Table:');
    disp(resultsTable);

    % Optionally, display individual outputs
    disp('OLS Coefficients (beta):');
    disp(beta);

    disp('Newey-West HAC Covariance Matrix:');
    disp(HAC_cov);

    disp('Newey-West Corrected Standard Errors:');
    disp(standardErrorsHAC);

    disp('t-Statistics:');
    disp(tStats);
end
