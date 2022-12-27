%Gianluca Galvagni 5521188
clearvars
clc
close all

%import the data set in table
mtcarsdatafile = readtable("mtcarsdata.csv");
turkish_se_SP500vsMSCIfile = readtable("turkish-se-SP500vsMSCI.csv");


%% 1) One-dimensional problem without intercept on the Turkish stock exchange data

Turkish_data = table2array(turkish_se_SP500vsMSCIfile);
%check if there are Nan values and change these to 0
Turkish_data(isnan(Turkish_data)) = 0;

X_turkish = Turkish_data(:,1);
T_turkish = Turkish_data(:,2);

%calculate the least square to the linear regression problem
W = ((sum(X_turkish.*T_turkish,'omitnan')) / (sum(X_turkish.^2)));

%plot the result
figure
scatter(X_turkish,T_turkish)
hold on;
grid on;

%i'm going to take the min and max value of the data setY_turkish
RopeX = [min(X_turkish),max(X_turkish)];
RopeT = W * RopeX;
plot(RopeX,RopeT)
xlabel("Standard and Poor's 500 return index");
ylabel("MSCI European index");
title("TASK 2) Least suqare solution with whole data set:");


%% 2) Compare graphically the solution obtained on different random subsets (10%) of the whole data set

%decide the % to remove from the principal data set
PercentualSubsets = 10;
New_valueData = floor(length(X_turkish) * (PercentualSubsets/100));

%choose the colors and simbols up to the graphic
color = {'b','r','g','c','k',''};
size = {10,15,20,25,30,35};
%choose the numbers of data set without the PercentualSubsets to plot
figure
Number_subset = 6;
while(Number_subset > 5 || Number_subset < 1)
    PromptNumbers = "How many data set (different) do you want?\nNB: NO more than 5 --> ";
    Number_subset = input(PromptNumbers);
end
%plot the graphics with the corrispective lines
for i=1:Number_subset
    allIndices = randperm(length(X_turkish));
    randomSubset = allIndices(1:New_valueData);
    
    x = Turkish_data(randomSubset,1);
    t = Turkish_data(randomSubset,2);

    Cs = (sum(x.*t)) / (sum(x.^2));

    scatter(x,t,size{i},color{i})
    hold on
    RopeX = [min(x),max(x)];
    RopeT = Cs * RopeX;
    plot(RopeX,RopeT,color{i})
end

grid on
xlabel("Standard and Poor's 500 return index");
ylabel("MSCI European index");
titlestring = "TASK 2) Least square solutions with " + Number_subset + " differents sub-data set";
title(titlestring);


%% 3) One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

cars_data = mtcarsdatafile{:,2:5};

X_carsWeight = cars_data(:,4);
T_carsMpg = cars_data(:,1);

%calcolate the values W0 and W1 to do the plot
W1 = ((sum((X_carsWeight - mean(X_carsWeight)).*(T_carsMpg - mean(T_carsMpg)))) / (sum((X_carsWeight - mean(X_carsWeight)).^2)));
W0 = mean(T_carsMpg) - W1 * mean(X_carsWeight);

figure
scatter(X_carsWeight,T_carsMpg)
hold on
grid on
Xweight = [min(X_carsWeight),max(X_carsWeight)];
Tmpg = W1.*Xweight + W0;
plot(Xweight,Tmpg)
xlabel("Car weight (lbs/1000)");
ylabel("Fuel efficiency (mpg)");
title("TASK 2) Motor Trends survey: Car mpg as a function of weight");


%% 4) Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)

%like the point 3 i am gonna choose the same plot but with all database. 
% t for the mpg and the x for the rests of data

t = cars_data(:,1);
x = cars_data(:,2:4);

%now we have to calculate 4 parameters W0,W1,W2,W3

%I have to add a column in x for can calculate the 4 parameters
a(1:length(x),1:1) = 1;
x = [a x];
%calculate Wn and print the result
Wvalues= pinv(x) * t;
fprintf("\n");
for i=1:length(Wvalues)
    fprintf('W%s = %f\n',(num2str(i)-1),Wvalues(i));
end
%NB pinv -> pseudoinverse


%% TASK 3
%% 5) Re-run 1,3 and 4 from task 2 using only 5% of the data.

PercentualUsable = 0;
fprintf("\n");
%here you can chose the % to split the data set
while(PercentualUsable < 1 || PercentualUsable > 99)
    ProptPercentual = "Choose the PERCENTUAL of your data set.\nNB: Number between 1 and 99 --> ";
    PercentualUsable = input(ProptPercentual);
end

t = 0;
fprintf("\n");
%chose how many differents data set you want to create to do a summary for
%calculate the MSE in the end
while(t < 1)
    PromptJ = "Choose the NUMBER of differents data set.\nNB: Number bigger than 1 -->";
    t = input(PromptJ);
end
J_MSEturkish_bar(1:t,1:2) = 0;
J_MSEcars_bar(1:t,1:2) = 0;
J_MSEcarsbig_bar(1:t,1:2) = 0;

for j=1:t

    %3.1 | part of the task 2 number 1
    New_valueDataTurkish = floor(length(X_turkish) * (PercentualUsable/100));
    allIndices = randperm(length(X_turkish));
    randomSubsetTurkish = allIndices(1:New_valueDataTurkish);
        
    xmin = Turkish_data(randomSubsetTurkish,1);
    tmin = Turkish_data(randomSubsetTurkish,2);    
    
    %calculate the least square to the linear regression problem
    W_min = ((sum(xmin.*xmin)) / (sum(xmin.^2)));

    for i=1:length(tmin)
        ymin(i,1) = W_min;
    end
    
    %3.3 | part of the task2 number 3
    New_valueData_mincar = round(length(X_carsWeight) * (PercentualUsable/100));
    allIndicescar = randperm(length(X_carsWeight));
    randomSubset_mincar = allIndicescar(1:New_valueData_mincar);
    
    x_minCars = cars_data(randomSubset_mincar,4);
    t_minCars = cars_data(randomSubset_mincar,1);
    W1min = ((sum((x_minCars - mean(x_minCars)).*(t_minCars - mean(t_minCars)))) / (sum((x_minCars - mean(x_minCars)).^2)));
    W0min = mean(t_minCars) - W1min * mean(x_minCars);

    for i=1:length(t_minCars)
        y_minCars(i,1)= W0min + W1min * x_minCars(i,1);
    end
    
    %3.4 | part of the task2 number 4
    tmincars = cars_data(randomSubset_mincar,1);
    xmincars = [cars_data(randomSubset_mincar,2) cars_data(randomSubset_mincar,3) cars_data(randomSubset_mincar,4)];
    b(1:length(xmincars(:,1)),1:1) = 1;
    xmincars = [b xmincars];
    Wvaluesmin= pinv(xmincars) * tmincars;
    
    fprintf("\n%d) With only %d PERCENT of data set:\n--> ",j,PercentualUsable);
        for i=1:length(Wvaluesmin)
            fprintf('W%s = %f ; ',(num2str(i)-1),Wvaluesmin(i));
        end
            
        for i=1:length(tmincars)
            ymincars(i,1) = Wvaluesmin(1,1) + Wvaluesmin(2,1) * xmincars(i,2) + Wvaluesmin(3,1) *  xmincars(i,3) + Wvaluesmin(4,1) * xmincars(i,4);
        end

    %% 6) Compute the objective (mean square error) on the training data
    
    %mean sqaure for the first traing set (i called it "min")
    MSEturkish_min = immse(ymin,tmin);
    
    MSEcars_min = immse(y_minCars,t_minCars);

    MSEcarsbig_min = immse(ymincars,tmincars); 

    %% 7) Compute the objective of the same models on the remaining 95% of the data.

    %calculate the second training set (i callled it "max")
    %3.1
    xmax = Turkish_data(:,1);
    xmax(randomSubsetTurkish,:) = [];
    tmax = Turkish_data(:,2);
    tmax(randomSubsetTurkish,:) = [];
    W_max = ((sum(xmax.*xmax)) / (sum(xmax.^2)));
    for i=1:length(tmax)
        ymax(i,1) = W_max;
    end
    %3.3
    x_maxCars = cars_data(:,4);
    x_maxCars(randomSubset_mincar,:) = [];
    t_maxCars = cars_data(:,1);
    t_maxCars(randomSubset_mincar,:) = [];
    W1max = ((sum((x_maxCars - mean(x_maxCars)).*(t_maxCars - mean(t_maxCars)))) / (sum((x_maxCars - mean(x_maxCars)).^2)));
    W0max = mean(t_maxCars) - W1max * mean(x_maxCars);
    for i=1:length(t_maxCars)
        y_maxCars(i,1)= W0max + W1max * x_maxCars(i,1);
    end
    %3.4
    xmaxCars = cars_data(:,2:4);
    xmaxCars(randomSubset_mincar,:) = [];
    tmaxCars = cars_data(:,1);
    tmaxCars(randomSubset_mincar,:) = [];
    c(1:length(xmaxCars(:,1)),1:1) = 1;
    xmaxCars = [c xmaxCars];
    Wvaluesmax= pinv(xmaxCars) * tmaxCars;
    %print the Ws of the second traing set to see the different between the
    %first and the second
    fprintf("\n%d) With only %d PERCENT of data set:\n--> ",j,(100-PercentualUsable));
        for i=1:length(Wvaluesmax)
            fprintf('W%s = %f ; ',(num2str(i)-1),Wvaluesmax(i));
        end
        fprintf("\n");

        for i=1:length(tmaxCars)
            ymaxcars(i,1) = Wvaluesmax(1,1) + Wvaluesmax(2,1) * xmaxCars(i,2) + Wvaluesmax(3,1) *  xmaxCars(i,3) + Wvaluesmax(4,1) * xmaxCars(i,4);
        end


    %mean sqaure for the second traing set
    MSEturkish_max = immse(ymax,tmax);
    
    MSEcars_max = immse(y_maxCars,t_maxCars);

    MSEcarsbig_max = immse(ymaxcars,tmaxCars);

    %put the MSE min and MSE max in the matrix for plot after
    J_MSEturkish_bar(j,1) = MSEturkish_min;
    J_MSEturkish_bar(j,2) = MSEturkish_max;

    J_MSEcars_bar(j,1) = MSEcars_min;
    J_MSEcars_bar(j,2) = MSEcars_max;

    J_MSEcarsbig_bar(j,1) = MSEcarsbig_min;
    J_MSEcarsbig_bar(j,2) = MSEcarsbig_max;
end

%create the treeh histrogram one for each case
%3.1
figure
histogram(J_MSEturkish_bar(:,1),10)
hold on
grid on
histogram(J_MSEturkish_bar(:,2),10)
xlabel("Objective function value");
ylabel("Number of occurrences");
title("TASK 3) Histogram of Mean Square Error values for " + t + " different TURKISH data subsets:");
Promptmin = PercentualUsable + "% of data set";
Promptmax = 100-PercentualUsable + "% of data set";
legend(Promptmin,Promptmax)
%3.3
figure
histogram(J_MSEcars_bar(:,1),10)
hold on
grid on
histogram(J_MSEcars_bar(:,2),10)
xlabel("Objective function value");
ylabel("Number of occurrences");
title("TASK 3) Histogram of Mean Square Error values for " + t + " different MTCARS data subsets:");
Promptmin = PercentualUsable + "% of data set";
Promptmax = 100-PercentualUsable + "% of data set";
legend(Promptmin,Promptmax)
%3.4
figure
histogram(J_MSEcarsbig_bar(:,1),10)
hold on
grid on
histogram(J_MSEcarsbig_bar(:,2),10)
xlabel("Objective function value");
ylabel("Number of occurrences");
title("TASK 3) Histogram of Mean Square Error values for " + t + " different ALL data into MTCARS data subsets:");
Promptmin = PercentualUsable + "% of data set";
Promptmax = 100-PercentualUsable + "% of data set";
legend(Promptmin,Promptmax)