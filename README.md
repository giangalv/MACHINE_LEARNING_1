# MACHINE_LEARNING_1

# Lab 1: Naive Bayes classifier

> Task 1: Data preprocessing 


> Task 2: Build a naive Bayes classifier


> Task 3: Improve the classifier with Laplace (additive) smoothing


> Think further

## Task 1: Data preprocessing

Here you have to download data and prepare them for being used with a Matlab program.

Download the Weather data set and the Weather data description.

Remark: For use with Matlab it is convenient to convert all attribute values into integers>=1. So they can be used to index matrices. You can do so by either using a plain text editor, or reading the data into a spreadsheet table and then saving it back (remember to "save as" a plain text file or at most a .csv file, not a spreadsheet file).

Remark: For use with Python, you can read the nominal values as strings. Python has a data type called dictionary that accepts any type (including strings) as keys and as values.

You could work with the numpy library alone. However many operations, like reading data files or splitting data into subsets, are easier if you use the Pandas library.

Remark: The data set is so small that you can do the calculations by hand to compare with your code, and check if it is correct.

Remark: In statistical jargon, nominal or categorical variables don't have "values" but "levels".

## Task 2: Build a naive Bayes classifier
Here you have to create a program (a Matlab function for instance) that takes the following parameters:

a set of data, as a n x (d+1) matrix, to be used as the training set
another set of data, as a m x c matrix, to be used as the test set
The program should:

Check that the number c of columns of the second matrix is either d + 1 or d, i.e. either the same as the number of columns of the first matrix or one less column (the last column, the target class, may be present or absent)
Check that no entry in any of the two data sets is <1
Train a Naive Bayes classifier on the training set (first input argument), using its last column as the target
Classify the test set according to the inferred rule, and return the classification obtained
If the test set has a column number d+1, use this as a target, compute and return the error rate obtained (number of errors / m)
Write a script that, without the user's intervention:

loads the weather data set (already converted in numeric form)
splits it into a training set with 10 randomly chosen patterns and a test set with the remaining 4 patterns
calls the naive Bayes classifier and reads the results
prints the results: classification for each pattern of the test set, corresponding target if present, error rate if computed.
Hint: From the slides, you see that probability = frequency = number/totalnumber.

What may not be immediately clear from the formulas (because it is implicit in the definition of a probability mass function) is that for each variable you have to compute a probability for each possible value. This is the probability mass function that is used in the formulae. It gives the probability for each value (conditioned to the class, i.e. a likelihood; but this does not matter for the calculations).

Explicitly in pseudocode, what you have to compute (and store in a matrix or in several matrices) is:

for each class c
  for each variable x
    for each possible value v for variable x
        the number of instances of class c that have variable x == value v...
        ...divided by the number of instances of class c
and this will be our estimate of the likelihood P(x=v | c)

Remarks: The classifier should be programmed in such a way to be suitable for the following situations:

other data sets of different cardinalities (n, m) and dimensionality (d)
test sets not including a target
test sets including attribute values that were not in the training set.
In the last case the program should issue an error and discard the corresponding pattern, because if you use that value to index a matrix it will likely go out of bounds (e.g., if you computed probabilities for values 1, 2 and 3 and you get value 4 in the test).

Hint: Use the slides (ML 2) to implement the classifier. Note that all attributes are categorical, but some are non-binary having more than 2 levels.

Hint: If you are not strong in programming, don't fear: it is simpler than it seems. Programs, as well as their specifications and requirements, are easier to write than to read.

In any case it is advisable to start with a core that does just part of what is required, for instance just load a data set and compute probabilities; and then add functionalities gradually in more iterations. (This even has a name: agile software development, a type of continual improvement process.) A good practice in this multiple iteration approach is to save each version of the program so as to be able to go back to a working one if you do something wrong.

## Task 3: Make the classifier robust to missing data with Laplace (additive) smoothing
You will observe that, with such a small data set, some combinations that appear in the test set were not encountered in the training set, so their probability is assumed to be zero. When you multiply many terms, just one zero will make the overall result be zero.

In general, it may be the case that some values of some attribute do not appear in the training set, however you know the number of levels in advance.

An example is binary attributes (e.g., true/false, present/absent...). You know that attribute x can be either true or false, but in your training set you only have observations with x = false. This does not mean that the probability of x = true is zero!!!

To deal with this case, you should introduce a kind of prior information, which is called additive smoothing or Laplace smoothing.

Laplace smoothing -- Suppose you have random variable x taking values in { 1, 2, ..., v } (v possible discrete values). What is the probability of observing a specific value i ?

We perform N experiments and value i occurs ni times. Then:

Without smoothing (simple frequency) the probability of observing value i is given by P(x = i) = n_i/N
With Laplace smoothing, the probability of observing value i is given by P(x = i) = (n_i+a)/(N+av) where a is a parameter (see below)
NOTE: Smoothing is NOT a way to improve the performance of an existing classifier. It is a way to solve a problem of missing data, that would prevent you from building a classifier. You should not expect a better performance from a classifier with smoothing if the version without smoothing already works!

You have to change your code in 2 ways:

1) In the data preparation step, add the information about the number of levels. This means that for each data column you should add the number of possible different values for that column.

In the case of the weather data, this can be a list (or vector) [ 3, 3, 2, 2 ]. You can add this list as a first row in all the data sets (training and test), to be interpreted with this special meaning.

2) When you compute probabilities, you introduce Laplace smoothing in your formulas by adding some terms that take into account your prior belief. Since you don't know anything, your prior belief is that all values are equally probable. In this case, Laplace smoothing gives probability = 1/2 (for binary variables) or more generally probability = 1/v (for variables with v possible values). This is why the number of possible values must be known (modification 1).

In formulae, you should replace:

P(attribute x = value y | class z) =
   (number of observations in class z where attribute x has value y) / (number of observations in class z)
with the following:

P(attribute x = value y | class z) =
   ((number of observations in class z where attribute x has value y) +a) / ((number of observations in class z) +av)
where:

v is the number of values of attribute x, as above;
you can use a = 1. As a further refinement, you can experiment with a > 1 (which means "I trust my prior belief more than the data") or with a < 1 (which means "I trust my prior belief less than the data")
## Think further
If you have large data sets, you may incur in numerical errors while multiplying a lot of frequencies together. A solution is to work with log probabilities, by transforming logarithmically all probabilities and turning all multiplications into sums.
How would you proceed if the input variables were continuous? (Hint: variable values are used to compute probabilities by counting, but theory tells us that probabilities may be obtained by probability mass functions directly if they are known analytically.)
You can experiment with the continuous variable case using the Iris data set (with its description). The four features can be made binary by (1) computing the average of each and (2) replacing each individual value with True or 1 if it is above the mean and False or 0 if it is below. You can use the median in place of the average.

# Lab2: Linear regression

> Task 1: Get data


> Task 2: Fit a linear regression model


> Task 3: Test regression model

## Task 1: Get data
Download the turkish-se-SP500vsMSCI and the mtcarsdata-4features data sets.

Do what is necessary to make them readable in Matlab, for instance with the load function or the csvread function

Original sources of the data:

The Turkish stock exchange data can be downloaded from the U.C.I. Machine Learning Repository along with many others.
The MT cars data are available as the command "mtcars" in the (open source) R statistical/data analysis language and environment, along with many other example data sets Here is the documentation.

## Task 2: Fit a linear regression model
Using the slides, reproduce the examples that you have seen during the lectures:

One-dimensional problem without intercept on the Turkish stock exchange data
Compare graphically the solution obtained on different random subsets (10%) of the whole data set
One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight
Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)
Remark: A random set of m unique indices from 1 to N is obtained in Matlab as follows:

allIndices = randperm(N);
randomSubset = allIndices(1:m)
or in one line:

randomSubset = randperm(N,m)
In python with import numpy as np it would be just one line:

randomSubset = np.random.permutation(N)[:m]
Remark: To make differences more visible, rather than random sets you can select observations from different ends of the data set, i.e. from the beginning and from the end.

The idea is that, since the data are collected across time (a multidimensional time series), data collected in similar periods may be more similar than data collected from the beginning and the end of the whole period, which is one year.

As for random data, these will sample the whole set and will likely contain some instances from the whole period.  What we are capturing with regression is the average behaviour of a data set. It is very likely that the average behaviour of two random data subsets will be similar to the average behaviour of the whole set; you may try it to convince yourself experimentally.

## Task 3: Test regression model
Re-run 1,3 and 4 from task 2 using only 5% of the data.

Compute the objective (mean square error) on the training data

Compute the objective of the same models on the remaining 95% of the data.

Repeat for different training-test random splits, for instance 10 times. Suggestion: write code for this job, don't waste time repeating manually. Matlab scripts are done for that.

Show the results (using a graph or a table of values) and comment.

# Lab 3: kNN classifier

> Task 1: Obtain a data set


> Task 2: Build a kNN classifier


> Task 3: Test the kNN classifiers

##Task 1: Obtain a data set
Download the mnist data set. The zipfile contains, in separate files, a training set, corresponding labels, a test set, corresponding labels, two Matlab functions to load data and labels, and one flexible function to load the data named loadMNIST.

The first two functions are there just because they are distributed along with the data. For this lab assignment you don't need them; you will just use loadMNIST. Read its synthetic documentation by typing:

help loadMNIST
Python solution: The machine learning library keras, part of tensorflow, contains ready-made functions. Just do the following:

from tensorflow.keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
The data represent 70 000 handwritten digits in 28x28 greyscale images, already split into a 60K-image training set and a 10K-image test set, and are a standard benchmark for machine learning tasks.

## Task 2: Build a kNN classifier
Here you have to create a program (a Matlab function for instance) that takes the following parameters (input arguments):

a set of data, as a n x d matrix, to be used as the training set
a corresponding column (a n x 1 matrix) of targets, i.e., class labels
another set of data, as a m x d matrix, to be used as the test set
an integer k
OPTIONALLY, another set of data, as a m x 1 matrix, to be used as the test set ground truth (class labels)
Remark: Note that the requirement here asks for a separate target column for both the training set and (optionally) the test set. This is one of the many possible forms in which data may be organised.

Remark: The optional column is given as the last argument; this simplifies coding. Many programming languages allow for optional arguments or default values for arguments that were not specified when calling the function. Arguments are indicated with a name even when calling the function: for instance in Python a possible syntax is knn(mydata = dataset). In this way they can be positioned in any order.

In Matlab this mechanism does not exist per se, but there is a convention which is to use "Name-Value pairs". These are arguments that are interpreted in pairs. The first element, always a string, is the name of the argument; the second one is the corresponding value to set. You may have used this to set some properties of graphs with the plot function. There are functions that make it easier to use this convention.

However, to simplify our lives, we choose to have a fixed argument ordering, and to place the optional argument last (since it is just one). In this way, we can simply check the number of argument received (built-in variable nargin)to see whether the last one is present or not.

The program should:

Check that the number of arguments received (nargin) equals at least the number of mandatory arguments
Check that the number of columns of the second matrix equals the number of columns of the first matrix
Check that k>0 and k<=cardinality of the training set (number of rows, above referred to as n)
Classify the test set according to the kNN rule, and return the classification obtained
If the test set has the optional additional column (nargin == n.mandatory + 1), use this as a target, compute and return the error rate obtained (number of errors / m)
Hint: Use the slides to implement the classifier.

## Task 3: Test the kNN classifier
Use the MNIST character recognition data.

Compute the accuracy on the test set

on 10 tasks: each digit vs the remaining 9 (i.e., recognize whether the observation is a 1 or not; recognize whether it is a 2 or not; ...; recognize whether it is a 0 or not)
for several values of k, e.g., k=1,2,3,4,5,10,15,20,30,40,50 (you can use these, or a subset of these, or add more numbers, and you can also implement the rule: "k should not be divisible by the number of classes," to avoid ties).
Provide data or graphs for any combination of these parameters (e.g. recognize 1 with k=1,2,3,4...; recognize 2 with k=1,2,3,4...; and so on).

## ADDENDUM

Due to the possibly excessive length of running the experiments, you may want to subsample the whole training set (60K instances) by taking, for example, a random 10% of the data (6K instances).

If you do so, the result will depend on the random sampling; therefore, it is necessary to do some statistics over some repeated experiments (for instance 10 different random subsamplings).

For each of the experiments above, compute a confusion matrix, and then from it classification quality indexes. Provide an indication of typical value (e.g. an average, or a median) and an appropriate measure of spread (e.g. a standard deviation, or an interval between two relevant percentiles). Summarise these in appropriate tables. 

