import pandas as pd;
import numpy as np;
import math;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.neighbors import KNeighborsClassifier;
import matplotlib.pyplot as plt

#Question 14
#___________

#Part A
df = pd.read_csv('Auto.data', sep='\s+');
df_new = df.assign(mpg01 = (df.mpg > df.median(numeric_only = True)['mpg']));
df_new["mpg01"] = df_new["mpg01"].replace({True: 1, False: 0});
#Remove incomplete datapoints
lengthData = len(df_new['mpg'])
list_Indices = np.array([i for i in range(lengthData)])
excludedRows = df_new.loc[df_new['horsepower'] == '?'].index.values
regressionVals = np.setdiff1d(list_Indices, excludedRows);
df_new2 = df_new.loc[regressionVals];
df_new2["horsepower"] = pd.to_numeric(df_new2["horsepower"])
#Part B
numeric_column_List = ['displacement', 'horsepower', 'weight', 'acceleration'];
categorical_column_List = ['cylinders', 'year', 'origin'];
for column in df_new2:
    if column in numeric_column_List:
        plt.scatter(df_new2[column], df_new2['mpg01']);
        plt.title(column + " vs. mpg01");
        plt.xlabel(column);
        plt.ylabel('mpg01');
        plt.show();
        plt.clf();
    elif column in categorical_column_List:
        df_new2[column], df_new2['mpg01'];
        column_Vals = [[],[]];
        for index, row in df_new2.iterrows():
            column_Vals[row['mpg01']].append(row[column])
        plt.boxplot(column_Vals, labels = ['mpg01 = 0', 'mpg01 = 1']);
        plt.title(column + " vs. mpg01");
        plt.show();
        plt.clf();

#Part C
x_train_set = df_new2.sample(frac = 0.8, random_state = 200);
x_test_set = df_new2.drop(x_train_set.index);
y_train_set = x_train_set.pop('mpg01');
y_test_set = x_test_set.pop('mpg01');
x_train_array = np.transpose(np.array([list(x_train_set['cylinders']),list(x_train_set['horsepower']),list(x_train_set['acceleration']),list(x_train_set['weight']),list(x_train_set['year'])]));
y_train_array = np.array(y_train_set);
x_test_array = np.transpose(np.array([list(x_test_set['cylinders']),list(x_test_set['horsepower']),list(x_test_set['acceleration']),list(x_test_set['weight']),list(x_test_set['year'])]));
y_test_array = np.array(y_test_set);


#part f
logisticRegr = LogisticRegression(max_iter = 10000);
logisticRegr.fit(x_train_array, y_train_array);
predictions = logisticRegr.predict(x_test_array);
Error_Rate = 0.0;
for i in range(len(predictions)):
    if predictions[i] != y_test_array[i]:
        Error_Rate = Error_Rate + 1.0;
Error_Rate = Error_Rate/len(predictions);

#part h
Error_Array = [];
neighbors_Num = []
for i in range(1,61):
    Error_KNN = 0.0
    neighbors_Num.append(i)
    kNN = KNeighborsClassifier(n_neighbors = i);
    kNN.fit(x_train_array, y_train_array);
    predictionsKNN = kNN.predict(x_test_array);
    for i in range(len(predictionsKNN)):
        if predictionsKNN[i] != y_test_array[i]:
            Error_KNN = Error_KNN + 1.0;
    Error_Array.append(Error_KNN/len(predictionsKNN));
plt.scatter(neighbors_Num, Error_Array);
plt.xlabel('Number of Neighbors');
plt.xticks(ticks = [5*i for i in range(int(len(neighbors_Num)/5))])
plt.ylabel('Test Error Rate');
plt.title('Error Rate vs. number of Neighbors in KNN on Auto dataset');
plt.show();
plt.clf();




#Question 16
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
data_Bos_new = data_Bos.assign(CRIM01 = data_Bos.CRIM > data_Bos.median(numeric_only = True)['CRIM']);
data_Bos_new['CRIM01'] = data_Bos_new['CRIM01'].replace({True: 1, False: 0});
Logistic_Test_Error_Rate = [];
KNN_Error_Rate_Min = [];
neighbors_array = [];
###Generating 20 random datasets
for i in range(16):
    x_train_set = data_Bos_new.sample(frac = 0.8);
    x_test_set = data_Bos_new.drop(x_train_set.index);
    y_train_set = x_train_set.pop('CRIM01');
    y_test_set = x_test_set.pop('CRIM01');
    x_train_set.pop('CRIM');
    x_test_set.pop('CRIM');
    x_train_array = x_train_set.to_numpy();
    x_test_array = x_test_set.to_numpy();
    y_train_array = y_train_set.to_numpy();
    y_test_array = y_test_set.to_numpy();
    ###Logisitic Regression
    logisticRegr.fit(x_train_array, y_train_array);
    predictions = logisticRegr.predict(x_test_array);
    logit_error_rate = 0.0
    for l in range(len(predictions)):
        if predictions[l] != y_test_array[l]:
            logit_error_rate = Error_Rate + 1.0;
    logit_error_rate = logit_error_rate/len(predictions);
    Logistic_Test_Error_Rate.append(logit_error_rate);
    #####K-Nearest Regression 1-10 neighbors
    Error_Array = [];
    lim_neighbors = 20;
    neighbors_array = [];
    colors_array = ['red', 'blue','yellow', 'black', 'green', 'orange', 'purple', 'brown', 'pink','gray','beige','coral', 'darkgoldenrod','darkolivegreen', 'lavender','maroon'];
    for j in range(1,lim_neighbors + 1):
        neighbors_array.append(j);
        Error_KNN = 0.0;
        kNN = KNeighborsClassifier(n_neighbors = j);
        kNN.fit(x_train_array, y_train_array);
        predictionsKNN = kNN.predict(x_test_array);
        for k in range(len(predictionsKNN)):
            if predictionsKNN[k] != y_test_array[k]:
                Error_KNN = Error_KNN + 1.0;
        Error_Array.append(Error_KNN/len(predictionsKNN));
    plt.plot(neighbors_array, Error_Array, color = colors_array[i]);
    KNN_Error_Rate_Min.append(min(Error_Array));
#print("Logistic Regression Error Array: ", Logistic_Test_Error_Rate);
#print("KNN Error Array: ", KNN_Error_Rate_Min);
plt.xlabel('Number of Neighbors');
plt.ylabel('Test Error');
plt.title('KNN - Test Error vs Num_Neighbors, Random Samples, Boston data');
plt.xticks(neighbors_array);
plt.show();
plt.clf();
plt.boxplot([Logistic_Test_Error_Rate, KNN_Error_Rate_Min], labels = ['Logit Test Error','KNN Test Error']);
plt.title('Comparison of Test Error - Minimum from KNN vs. Logistic Regression');
plt.show();
plt.clf();
