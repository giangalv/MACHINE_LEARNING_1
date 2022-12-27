%Gianluca Galvagni 5521188
clearvars
clc
close all

%% PREPARE DATA

[train, labelTrain] = loadMNIST(0);
[test, labelTest] = loadMNIST(1);

%adding last column as target
training = [train, labelTrain];
testing = [test, labelTest];

NumTrainTot = length(training(:,1));
NumTestTot = length(testing(:,2));

%choose only a part of the data
numTrain = 6000; %number of test from training | min=max(k) & max=60000
numTest = 100; %number of test from testing | max=10000
numClass = 10; %number of classes

%Take random row for train and test
trainingSub = training(randi(NumTrainTot, [1,numTrain]), :);
testingSub = testing(randi(NumTestTot,[1,numTest]), :);

%% CLASSIFIER WITH DIFFERENT K VALUE

% I put here k values  
kNeighbours = [1:5, 10:5:20, 30:10:50];

numNeighbours = numel(kNeighbours);
out = zeros(numTest, numNeighbours);
err = zeros(1,numNeighbours);
for i=1:numNeighbours
    k = kNeighbours(i);
    [out(:,i), err(:,i)] = Galvagni_KNNClassifier(trainingSub, testingSub, k);
end

%% PLOT ERROR RATES

figure
h = bar(err);
h(1).FaceColor = 'r';
set(gca,'xticklabel',kNeighbours)
title("ERROR RATE (number of errors/number of test set rows");
ylabel("ERROR in percentual");
xlabel("Number of Neighbours");
grid on;

%% ONE - VSALL PROBLEM

%each column of vsAll matrix will refer to a class, each row has a
%observation of test set. if vsAll(i,j) == 1, then observation i is
%classified as clasj, otherwise it is 0 and the observation is not
%classified as clasj
vsAll = zeros(numTest, numClass, numNeighbours);
label = [1:numClass];

for i=1:numNeighbours
    for j=1:numTest
        for z=1:numClass
            if out(j,i) == label(z)
                vsAll(j,z,i) = 1;
            else
                vsAll(j,z,i) = 0;
            end
        end
    end
end

%% PLOT CLASSIFICATION for a VALUE 

%choose the value of k
kValue = 5;
figure
h = heatmap(vsAll(:,:,kValue));
prompt = "ONE - VSALL problems for each class, with " + kValue + " NEIGHTBOURS. FULL BLUE BOX means that the test has been classified as that class.";
title(prompt);
xlabel("CLASSES");
ylabel("TESTS");
h.ColorbarVisible = 'off';
%here you can see where the test is classified

%% CONFUSION MATRIX, ACCURACY, SENSITIVITY, SPECIFICITY, PRECISION, QUALITY INDEXES, AVERAGE and SPREAD

truePos = zeros(numNeighbours,numClass);
trueNeg = zeros(numNeighbours,numClass);
falsePos = zeros(numNeighbours,numClass);
falseNeg = zeros(numNeighbours,numClass);

for k=1:numNeighbours
    for class=1:numClass
        for ask=1:numTest
            if(testingSub(ask,end) == class)
                if(vsAll(ask,class,k) == 1)
                    truePos(k,class) = truePos(k,class) + 1;
                else
                    falseNeg(k,class) = falseNeg(k,class) + 1;
                end
            else
                if(vsAll(ask,class,k) == 1)
                    falsePos(k,class) = falsePos(k,class) + 1;
                else
                    trueNeg(k,class) = trueNeg(k,class) + 1;
                end
            end
        end
    end
end

Accuracy = (truePos + trueNeg) ./ (truePos + trueNeg + falsePos + falseNeg);
Sensitivity = truePos ./ (truePos + falseNeg);
Specificity = trueNeg ./ (trueNeg + falsePos);
Precision = truePos ./ (truePos + falsePos);

%Quality index
FoneIndex = (2 * (truePos.^2)) ./ (2 * truePos + trueNeg + falsePos);

Average = sum(FoneIndex);
%I consider only the class with a "good" K
goodClass = numNeighbours - sum(isnan(out(1,1:numNeighbours)));
Average = Average / goodClass;

Spread = std(FoneIndex);

%% PLOTS the 10 TASKS (10 classes)

leg{1} = "Accuracy";
leg{2} = "Sensitivity";
leg{3} = "Specificity";
leg{4} = "Precision";

prompt = "With K values: ";
for a=1:numNeighbours
    prompt = prompt + kNeighbours(a) + " ";
end

figure
sgtitle(prompt)
for p=1:6
    subplot(3,2,p)
    h = bar([Accuracy(:,p).*100, Sensitivity(:,p).*100, Specificity(:,p).*100,Precision(:,p).*100]);
    ylim([0 105])
    ytickformat('%g %%')
    xlabel("Number of Neighbours");
    h(1).FaceColor = 'r';
    h(2).FaceColor = 'g';
    h(3).FaceColor = 'b';
    h(4).FaceColor = 'y';    
    set(gca,'xticklabel',kNeighbours)
    title("Recognize " + num2str(p));
end
legend(h,leg)

figure
sgtitle(prompt)
for p=7:10
    subplot(3,2,(p-6))
    h = bar([Accuracy(:,p).*100, Sensitivity(:,p).*100, Specificity(:,p).*100,Precision(:,p).*100]);
    ylim([0 105])
    ytickformat('%g %%')
    xlabel("Number of Neighbours");
    h(1).FaceColor = 'r';
    h(2).FaceColor = 'g';
    h(3).FaceColor = 'b';
    h(4).FaceColor = 'y';    
    set(gca,'xticklabel',kNeighbours)
    title("Recognize " + num2str(p));
end
legend(h,leg)

figure
sgtitle("ADDENDUM")
for p=1:6
    subplot(3,2,p)
    h = bar(FoneIndex(:,p),'black');
    xlabel("Number of Neighbours: ")
    set(gca,'xticklabel',kNeighbours)
    title("Class " + num2str(p));
    yline(Average(1,p),'-.m','AVERAGE')
    yline(Spread(1,p),'-.c','SPREAD')
end

figure
sgtitle("ADDENDUM")
for p=7:10
    subplot(3,2,(p-6))
    h = bar(FoneIndex(:,p),'black');
    xlabel("Number of Neighbours: ")
    set(gca,'xticklabel',kNeighbours)
    title("Class " + num2str(p));
    yline(Average(1,p),'-.m','AVERAGE')
    yline(Spread(1,p),'-.c','SPREAD')
end
