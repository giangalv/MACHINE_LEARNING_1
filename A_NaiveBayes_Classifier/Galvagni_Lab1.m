%Gianluca Galvagni 5521188
% SET VALUES:
% 3  ->  sunny;     hot;   high;    true;   yes.
% 2  ->  overcast;  mild;  normal;  false;  no.
% 1  ->  rainy;     cold.
% Import the data set in table TABLE
filename = "1_NaiveBayesClassifier.csv";
W = readtable(filename);

Matrixset = table2array(W);
%check if there are values < 1 inside the matrixset
Error_M = 0;
Error_Matrix(:,:) = 0;
[Rowsmatrixset,Colsmatrixset] = size(Matrixset);
for r=1:Rowsmatrixset
    for c=1:Colsmatrixset
        if Matrixset(r,c)<1
            Error_M = Error_M + 1;
            Error_Matrix(Error_M,1) = r;
            Error_Matrix(Error_M,2) = c;
        end
    end
end
%print the reuslt of the errors' number < 1
if Error_M==0
    fprintf("NO numbers smaller than 1 inside the DATASET\n\n");
else
    fprintf("ERROR, %d numbers smaller than 1 inside the DATASET.\nThe wrong numbers are in the cells:\n",Error_M);
    fprintf('[%d,%d]\n', Error_Matrix.');
    fprintf("\n\n")
end

%ask to the customer to tell the number of columns with the max value 3 and
% the boolean columns
PromptThree = "How many datasets'columns have 3 like a max value?\nNB: Put these columns before the boolean columns value (on the left of your DATASET).\n--> ";
ColounswithThree = input(PromptThree);
PromptTwo = "\nHow many datasets'columns have a boolean values?\nNB: Count here also the last colunm with the 'answers'.\nNNB: 3=TRUE and 2=FALSE.\n--> ";
ColounswithBoolean = input(PromptTwo);
%create the vector to know the max values of all colunms
Max_Values(1:ColounswithThree) = 3;
Max_Values(ColounswithThree+1 : ColounswithThree + ColounswithBoolean - 1) = 2;

Error_D = 0;
Error_Data(:,:) = 0;
for r=1:Rowsmatrixset
    for c=1:ColounswithThree
        if Matrixset(r,c)>3
            Error_D = Error_D + 1;
            Error_Data(Error_D,1) = r;
            Error_Data(Error_D,2) = c;
        end
    end
    x = ColounswithThree + 1;
    z = ColounswithBoolean + ColounswithThree;
    for n=x:z
        %I decided to use only 2 and 3 like a boolean values 3=TRUE &
        %2=FALSE
        if Matrixset(r,n)~=2 && Matrixset(r,n)~=3
            Error_D = Error_D + 1;
            Error_Data(Error_D,1) = r;
            Error_Data(Error_D,2) = n;
        end
    end
end
%print the reuslt of the errors inside the dataset and the cells with the
%error
if Error_D~=0
    fprintf("ERROR, %d cells with a wrong value inside the DATASET.\nThe wrong values are in the cells:\n",Error_D);
    fprintf('[%d,%d]\n', Error_Data.');
    fprintf("\n\n")
else
    fprintf("NO ERRORS inside the DATASET\n\n");
end

%vector with 4 random numbers
randomvector = randperm(Rowsmatrixset,4); 
%sort of the randomvector to remove rows in order
randomvector = sort(randomvector);

%create the training set from the data set, i'm going to remove 4 rows
Trainingnset = table2array(W);
a = 0;
for r=1:4
    n = randomvector(r) - a;
    Trainingnset(n,:) = [];
    a = a + 1;
end

%create the test set from the data set without the rows' trainingset
Testset = table2array(W);
[~,index] = intersect(Matrixset,Trainingnset,'rows');
Testset(index,:) = [];

%create the Y for the Naive Bayes classifier from training set
[Rowstrainingset,Colstrainingset] = size(Trainingnset);
Y(1) = 0;
for r=1:Rowstrainingset
    Y(r,1) = Trainingnset(r,Colstrainingset);
end
%remove the Y from the training set
Trainingnset(:,Colstrainingset) = [];

%create the model for the Bayes
[Rowstrainingset_B,Colstrainingset_B] = size(Trainingnset);
%Calculate the P(Y) & P(N)
Play_Yes = 0;
Play_No = 0;
[Rowstrainingset_Y,~] = size(Y);
for r=1:Rowstrainingset_Y
    if Y(r,1)==3,1;
        Play_Yes = Play_Yes + 1;
    elseif Y(r,1)==2,1;
        Play_No = Play_No + 1;
    end
end
Pp_Yes = Play_Yes / Rowstrainingset_Y;
Pp_No = Play_No / Rowstrainingset_Y;

%watch the training set and count how many times the Y/N is correlated
% with the 4 variables(colunms)
P_Marginalizzation(1:3,1:Colstrainingset_B,1:2) = 0;
%Prob_X(:,:,1) for NO & Prob_X(:,:,2) for YES
for b=1:2
    for c=1:Colstrainingset_B
            for r=1:Rowstrainingset_B
                for v=1:3
                    if Y(r)==(b+1) && Trainingnset(r,c)==v
                        P_Marginalizzation(v,c,b) = P_Marginalizzation(v,c,b) + 1;                             
                    end
                end 
            end
    end
end

%create the likelihood matrix + Laplace smoothing 
Laplace_Value = 1;
P_Likelihood(1:3,1:Colstrainingset_B,1:2) = 0;
for b=1:2
    for c=1:Colstrainingset_B
        for v=1:3
            if b==1
                P_Likelihood(v,c,b) = (P_Marginalizzation(v,c,b) + Laplace_Value) / (Play_No + Laplace_Value * Max_Values(c));   
            elseif b==2
                P_Likelihood(v,c,b) = (P_Marginalizzation(v,c,b) + Laplace_Value) / (Play_Yes + Laplace_Value * Max_Values(c));
            end
        end
    end
end

%set to zero the likelihood matrix where you don't have to have the
%possibility to have cell != 0. For e.g. in boolean values
for b=1:2
    for c=ColounswithThree+1 : ColounswithThree+ColounswithBoolean-1
        P_Likelihood(1,c,b) = 0;
    end
end

%calculate the Marginalizzation matrix
P_Marginalizzation = rdivide(P_Marginalizzation,Rowstrainingset_B);

%create the table to confrot from the test table using the Bayes classifier
[RowsTestset,ColsTestset] = size(Testset);
P_Posterior(1:RowsTestset,1:3) = 1;
for r=1:RowsTestset
   for b=1:2
            if b==1
                for v=1:Colstrainingset_B
                    P_Posterior(r,b) = P_Posterior(r,b) * P_Likelihood(Testset(r,v),v,b);
                end
                P_Posterior(r,b) = P_Posterior(r,b) * Pp_No;
                for v=1:Colstrainingset_B
                    P_Posterior(r,b) = P_Posterior(r,b) * (1 / P_Marginalizzation(Testset(r,v),v,b));
                end
            elseif b==2
                for v=1:Colstrainingset_B
                    P_Posterior(r,b) = P_Posterior(r,b) * P_Likelihood(Testset(r,v),v,b);
                end
                P_Posterior(r,b) = P_Posterior(r,b) * Pp_Yes;
                for v=1:Colstrainingset_B
                    P_Posterior(r,b) = P_Posterior(r,b) * (1 / P_Marginalizzation(Testset(r,v),v,b));
                end
            end
    end
end

%normalize the Posterior matrix done + change Nan values and Inf values 
%with 0.1 in the Posterior matrix for don't have a 0 in the matrix and put
%likely to 100% 
for r=1:RowsTestset
    if isnan(P_Posterior(r,1))==true || isinf(P_Posterior(r,1))==true
        P_Posterior(r,1) = 0.1;
    end
    if isnan(P_Posterior(r,2))==true || isinf(P_Posterior(r,2))==true
        P_Posterior(r,2) = 0.1;
    end
    b = P_Posterior(r,1); 
    v = P_Posterior(r,2);
    P_Posterior(r,1) = b / (b + v);
    P_Posterior(r,2) = v / (v + b);
end

%check the error and print the results
Error = 0;
fprintf("Test: --  Result:\n");    
for r=1:RowsTestset
    %print the result of Test
    if Testset(r,ColsTestset)==3
         fprintf("YES   --  ");         
    elseif Testset(r,ColsTestset)==2
        fprintf("NO    --  ");
    end
    %print the result of Bayes
    if P_Posterior(r,1) == P_Posterior(r,2)
        fprintf("ERROR\n");
        P_Posterior(r,3) = 1;
    elseif P_Posterior(r,1) < P_Posterior(r,2)
        fprintf("YES\n");
        P_Posterior(r,3) = 3;
    elseif P_Posterior(r,1) > P_Posterior(r,2)
        fprintf("NO\n");
        P_Posterior(r,3) = 2;
    else
        fprintf("ERROR\n");
        P_Posterior(r,3) = 1;
    end

    if Testset(r,ColsTestset)~=P_Posterior(r,3)
        Error = Error + 1;
    end     
end
ErrorRate = Error / RowsTestset;
fprintf("Error rate = %.2f",ErrorRate);