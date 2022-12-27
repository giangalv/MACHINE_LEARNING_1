%Gianluca Galvagni 5521188
clearvars
clc
close all

%import the data set in table
winefile = uiimport("wine.data");
breastcancerfile = uiimport("breast-cancer-wisconsin.data");

%% WINE DATA
WINEdata = struct2array(winefile);
%traspost for the matlabs' function
WINEdata_trasp = WINEdata'; 
x_WINE = WINEdata_trasp(2:end,:);
t_WINE = WINEdata_trasp(1,:);

%% BREAST CANCER DATA
CANCERdata = struct2array(breastcancerfile);
%traspost for the matlabs' function
CANCERdata_trasp = CANCERdata';
x_CANCER = CANCERdata_trasp(1:end-1,:);
t_CANCER = CANCERdata_trasp(end,:);

%% TASK 1) Solve a Pattern Recognition Problem with a Neural Network

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
fprintf("START TASK1:\n")
exit = 'N';
while not(strcmp(exit,'y'))
    hiddenLayerSize = 0;
    while (hiddenLayerSize < 1 || hiddenLayerSize > 50)
        PromptLayerSize = "Choose the SIZE of a LAYER (number between 1 and 50): ";
        hiddenLayerSize = input(PromptLayerSize);
        fprintf("\n");
    end
    
    numberLayers = 0;
    while (numberLayers < 1 || numberLayers > 30)
        PromptNummberUnits = "Choose the NUMBER of LAYERS (number between 1 and 30): ";
        numberLayers = input(PromptNummberUnits);
        fprintf("\n");
    end

    arrayLayers(1:numberLayers) = hiddenLayerSize;
    net = patternnet(arrayLayers, trainFcn);

    numberData = 0;
    while (numberData ~= 1 && numberData ~= 2)
        PromptData = "Choose the Data:\n-> [1] for WINEdata;\n-> [2] for BREAST-CANCERdata.\n-> ";
        numberData = input(PromptData);
    end

    if (numberData == 1)
        x = x_WINE;
        t = t_WINE;
    elseif (numberData == 2)
        x = x_CANCER;
        t = t_CANCER;
    end

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,tr] = train(net,x,t);
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    
    % View the Network
    view(net)

    plotYoN = input("Do you want to plot the result?\n Y/N (you can print only with y/Y): ","s");
    plotYoN = lower(plotYoN);
    if strcmp(plotYoN,'y')
        % Plots
        % Uncomment these lines to enable various plots.
        figure, plotperform(tr)
        figure, plottrainstate(tr)
        figure, ploterrhist(e)
        figure, plotconfusion(t,y)
        figure, plotroc(t,y)
    end

    exit = input('Do you want to exit?\n Y/N (you can exit only with y/Y): ','s');
    exit = lower(exit);
end
 
%% TASK 2) Autoencoder
clearvars
clc
close all

%% Prepare data set

fprintf("\nSTART TASK2:\n");
% the second argument is to choose which class extract (10 for handwritten
% 0)
[trainA,labelTrainA] = loadMNIST(0,1);
[trainB,labelTrainB] = loadMNIST(0,7);
NtrainA = length(trainA(:,1));
NtrainB = length(trainB(:,1));

% put togheter observatio and label so with rand I extract random row with
% correspondent label
trainA = [trainA,labelTrainA];
trainB = [trainB,labelTrainB];

% choose the size of the subtrains
nTrain = 100;
trainSubA = trainA(randi(NtrainA, [1,nTrain]),:);
trainSubB = trainB(randi(NtrainB,[1,nTrain]),:);
training = [trainSubA(:,1:end-1);trainSubB(:,1:end-1)];

labelTrain = [trainSubA(:,end); trainSubB(:,end)];

% number of hidden units, 2 so I can plot learning into a 2d plot
nh = 2;
% myAutoencoder = trainAutoencoder(myData,nh);
% myEncodedData = encode(myAutoencoder,myData);
myAutoencoder = trainAutoencoder(training',nh);
myEncodedData = encode(myAutoencoder,training');

% PLOT THE RESULTS
plotcl(myEncodedData',labelTrain)
xlabel('Hidden unit 1');
ylabel('Hidden unit 2');
title(["Output of the autoencoder", newline, ...
    "with " + num2str(nTrain) + " instance of classes " + num2str(labelTrainA(1)) ...
    + " and " + num2str(labelTrainB(1))]);
legend("Class " + num2str(labelTrainA(1)), "Class " + num2str(labelTrainB(1)));


