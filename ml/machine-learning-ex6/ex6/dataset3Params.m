function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;




% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_list = [0.01, 0.01, 0.1, 0.3, 1, 3, 10, 30];
sigma_list =  [0.01, 0.01, 0.1, 0.3, 1, 3, 10, 30];
results = [0.0, 0.0];
min_cost = 9999;

for c = C_list
    for s = sigma_list
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        predictions = svmPredict(model, Xval);
        current_cost = mean(double(predictions ~= yval))
        if min_cost > current_cost
            min_cost = current_cost;
            results(1) = c
            results(2) = s
        endif
    endfor
endfor


C = results(1);
sigma = results(2);





% =========================================================================

end
