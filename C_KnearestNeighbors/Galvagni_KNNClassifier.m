function [out, err] = Galvagni_KNNClassifier(train,test,k)

%% Galvagni_KNNClassifier

%This function classifies each row of the test set according to KNN
%Classifier rule based on the train test. It also return the error if the
%true labels for the test are provided.

%INPUT:
%   train: as a (n x d) matrix, to be used as the training set
%   targets: a corresponding column (n x 1) vector of targets|class labels
%   set: as a (m x c) matrix, to be used as the test set, with c >= d-1
%   k: an integer 

%OUTPUT:
%   out: as a (m x 1) vector which classify the test data
%   err: 0->(min err) to 1->(max err) scalar find as 
%        (number of classifying errors / m)
%        if c == d - 1 err can not be compute and will be NaN 

%% 

[nRowTrain, nColTrain] = size(train);
[nRowTest, nColTest] = size(test);
nClasses = max(unique(train(:,end)));

%% Checking inputs

%Check that the number of arguments received (nargin) equals at least the number of mandatory arguments
if(nColTest < nColTrain - 1)
    error("Number of columns of test data must be at least number of column of the training data - 1");
end

%Check that the number of columns of the second matrix equals the number of columns of the first matrix
if(nColTest ~= nColTrain)
    error("Number of columns of test data must be the same number of columns of the training data");
end

%Check that k>0 and k<=cardinality of the training set (number of rows, above referred to as n)
if(k <= 0 || k > nRowTrain)
    error("Not a valid K");
end

%rule: "k should not be divisible by the number of classes"
if(mod(k,nClasses) == 0)
    fprintf("K: " + k + " should not be divisible by the number of classes\n");
    out = NaN;
    err = NaN;
    return
end

%% Classifier

%create the array out with 0 values
out = zeros(nRowTest,1);

for ask=1:nRowTest

    %EUCLIDIAN NORM to calculate the distance
    %vecnorm calculate norm on the row
    norms = vecnorm(train(:,1:end-1) - test(ask,1:(nColTrain-1)), 2, 2);
    n = zeros(k,1);

    %Computationally is more efficient with the min() k time if
    %k<log2(nRowtrain) becuase using min complexity is O(nk), instead
    %comlexity of sorting is O(nLog2(nRowtrain))
    if(k < log2(nRowTrain))
        for ki=1:k
            [~,n(ki)] = min(norms);
            norms(n(ki)) = Inf;
            %put inf value because for the next interraction this value
            %won't taken into account
        end
    else
        %ordering to find the neighbours
        [~, sortedIndex] = sort(norms);
        %choose the first k neighbours
        n = sortedIndex(1:k);
    end

    %take the most frequently value of the first k neigbours
    out(ask) = mode(train(n,end), 1);
end

%% Error
%if the train has more, i assume that the real target is in the nColTest
%column of the test data
err = NaN;

%i calcolate the error only if the columns of test are >= than the columns
%of the train
if nColTest >= nColTrain % i can compute th error
    %Nan value is out is considered an error
    %max error is 1
    err = (sum(out ~= test(:,nColTrain))) / nRowTest;
end
