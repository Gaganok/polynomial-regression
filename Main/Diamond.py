'''
Created on Dec 3, 2019

@author: Oleh Kepsha
@student_id: R00150485
'''

import numpy as np
import pandas as ps;
import matplotlib.pyplot as plt
import math

max_deg= 4; #Max_degree to be tested (0-3)
k = 4; # K_Fold split
n = 800; #Min Number of DataPoint to be able to process the set

#Task 1
def main():
    #Task 1
    data = ps.read_csv("diamonds.csv");
    
    #Types of cut, color and clarity
    cuts = data["cut"].unique();
    colors = data["color"].unique();
    clarities = data["clarity"].unique();
    
    #Extracting data points for each combination
    data_set, comb_set = combinations_datapoints(data, cuts, colors, clarities);
    
    #Spliting data sets into features and target
    # data_set is split in 5 combination > n datapoints. 
    #Each combination is split in 2 arrays, 1 for feature and 2 for target
    for i in range(len(data_set)):
        data_set[i] = (data_set[i][["carat", "depth", "table"]], data_set[i][["price"]])
     
    plt.close("all")
    
    for i in range(len(data_set)):
        data = np.array(data_set[i][0])
        target = np.array(data_set[i][1])
        target = np.reshape(target, len(target))
        
        print("Cut: " + comb_set[i][0] + " Color: " + comb_set[i][1] + " Clarity: " + comb_set[i][2]);
        #Task 6
        best_degree = k_fold_cross_val(data, target, k, max_deg)
        
        #Task 7
        #Estimate the model parameters for each dataset
        param_vector = regression(data, target, best_degree)
        #Calculate the estimated price for each diamond in the dataset using the estimated model parameters 
        prediction = calculate_model_function(best_degree, data, param_vector)
        #Plot the estimated prices against the true sale prices
        plt.plot(target,'ro', c='r')
        plt.plot(prediction,'ro', c='b')
        plt.show()
        
#Task 1
def combinations_datapoints(df, cuts, colors, clarities):
    data_set = []
    comb_mapping = []
    for cut in cuts:
        for color in colors:
            for clarity in clarities:
                data = df[(df["cut"] == cut) & (df["color"] == color) & (df["clarity"] == clarity)];
                length = len(data);
                
                #Counting the number of data-points in each combination
                #!!!!!! Uncomment to see !!!!!!!!!
                #print("Cut: " + cut + " Color: " + color + " Clarity: " + clarity + " Count/Total Amount: " + str(length));
                
                #Selecting only data sets with > n(800) data points
                if(length >= n):
                    comb_mapping.append((cut, color, clarity))
                    data_set.append(data);
                    
    return data_set, comb_mapping;
  
#Task 2
#function determines the correct size for the parameter vector
def size_parameter_vector(d):
    size = 0
    for n in range(d+1):
        for x1 in range(n+1): #Carat
            for x2 in range(n+1): #Depth
                for x3 in range(n+1): #Table
                    if x1+x2+x3==n:
                        size+=1
    return size

#Task 2
#function calculates the estimated target vector
def calculate_model_function(deg,data, p):
    result = np.zeros(data.shape[0])    
    index=0
    for n in range(deg+1):
        for i in range(n+1):
            for k in range(i+1):
                #included 3 polynomials for carat, depth and table
                result += p[index]*(data[:,0]**i)*(data[:,1]**(n-i))*(data[:,2]**(n-k))
                index+=1        
    return result

#Task 3
def linearize(deg,data, p0):
    #estimated target vector 
    f0 = calculate_model_function(deg,data,p0)
    #Jacobian matrix
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg,data,p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    return f0,J

#Task 4
def calculate_update(y,f0,J):
    #regularisation term
    l=1e-2
    #normal equation matrix
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    #residual 
    r = y-f0
    #normal equation system
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp            

#Task 5
def regression(data, target, deg):
    max_iter = 10
    #parameter vector of coefficients with zeros 
    param_vector = np.zeros(size_parameter_vector(deg))
    #procedure that alternates linearization and parameter update
    for i in range(max_iter):
        f0,J = linearize(deg,data, param_vector)
        dp = calculate_update(target,f0,J)
        param_vector += dp

    return param_vector

#Task 6
#Use the mean absolute price difference as measure of quality for the different model functions
def accuracy(target, prediction):
    distance = 0;
    for i in range(len(target)):
        distance += math.fabs(target[i] - prediction[i])
    return distance / len(target)

#Task 6
def k_fold_cross_val(data, target, k, degree):
    #put initial data into the list
    #-1 is invalid degree and really big number 10**10 to override with next available mean_distance
    best_degree = [-1, 10**10];
    
    #Number of indexes in a split
    split = (len(target)/k)
    for deg in range(degree):
        kf = k;
        deg_mean = 0;
        while kf > 0:
            #Picking index for splitting the data
            index = np.arange(int((kf-1) * split),int(kf * split))
            
            k_data = data[index]
            k_target = target[index]
            
            param_vector = regression(k_data, k_target, deg)
            prediction = calculate_model_function(deg, k_data, param_vector)
            #getting current spit data accuracy score (absolute distance between actual and predicted values)
            deg_mean += accuracy(k_target, prediction)
 
            kf-=1;
        #getting average accuracy score
        deg_mean /= k;
        #Compare current accuracy with the last best accuracy
        if(best_degree[1] > deg_mean):
            best_degree[0] = deg;
            best_degree[1] = deg_mean;
    print("Best fit degree: ", best_degree[0], " Mean Distance: " , best_degree[1])
    #Return best polynomial degree
    return best_degree[0]

main();