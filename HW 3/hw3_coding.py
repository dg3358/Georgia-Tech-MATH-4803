import pandas as pd;
import numpy as np;
import math;
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold;
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis;
from sklearn.naive_bayes import GaussianNB;
from sklearn.linear_model import LinearRegression;
import statsmodels.api as sm;
import matplotlib.pyplot as plt;
from patsy import dmatrix;


#Helper Functions
def MeanSquaredError(y_vals, y_hat):
    diff = y_vals - y_hat;
    diff_sq = np.transpose(diff)*diff
    mse = np.sum(diff_sq)/len(y_vals);
    return mse;

def Q9_helper(deg_polynomial):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
    dis_vals = data_Bos['DIS'].to_numpy();
    y_Vals = data_Bos['NOX'].to_numpy();
    X_Vals = [dis_vals];
    for i in range(1, deg_polynomial):
        curr_dis_vals = np.array(X_Vals[len(X_Vals) - 1]);
        next_dis_vals = np.transpose(curr_dis_vals)*dis_vals;
        X_Vals.append(next_dis_vals);
    X_Vals = np.array(X_Vals);
    X_Vals = np.transpose(X_Vals)
    column_Headers = ["dis^{}".format(j) for j in range(1, deg_polynomial + 1)];
    x_values = pd.DataFrame(data = X_Vals, columns = column_Headers);
    y_values = pd.DataFrame(data = y_Vals, columns = ['nox']);
    x_values = sm.add_constant(x_values);
    linReg = sm.OLS(y_values, x_values).fit();
    return linReg;

def Q9_helper_2(deg_polynomial, num_knots):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    quantiles = [i/(num_knots + 1) for i in range(1, num_knots + 1)];
    knot_Locations = [data_Bos.quantile(j)['DIS'] for j in quantiles];
    tup_Locations = tuple(knot_Locations);
    basis_x = dmatrix("bs(data_Bos.DIS, knots = tup_Locations, degree = deg_polynomial, include_intercept = False)", {"data_Bos.DIS": data_Bos['DIS']},return_type = 'dataframe');
    spl_mod = sm.GLM(data_Bos['NOX'], basis_x).fit();
    Predicted_outputs = spl_mod.predict(basis_x).to_numpy().flatten();
    y_Vals = data_Bos['NOX'].to_numpy().flatten();
    Resid_sum_squares = MeanSquaredError(Predicted_outputs, y_Vals)*len(Predicted_outputs);
    X = np.linspace(data_Bos.min()['DIS'], data_Bos.max()['DIS'], num = 1000);
    pred = spl_mod.predict(dmatrix("bs(X, knots = tup_Locations, degree = deg_polynomial, include_intercept = False)", {"X": X},return_type = 'dataframe'));
    return spl_mod, Resid_sum_squares, X, pred;



#Exercise 5 - Chapter 4, Question 14 (ctd)
#----------------
#creating mpg01 (from HW 2)
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

#split dataset into train, test
x_train_set = df_new2.sample(frac = 0.8, random_state = 200);
x_test_set = df_new2.drop(x_train_set.index);
y_train_set = x_train_set.pop('mpg01');
y_test_set = x_test_set.pop('mpg01');
#From HW 2, important paramaters selected: cylinders, horsepower, acceleration, weight, and year
x_train_array = np.transpose(np.array([list(x_train_set['cylinders']),list(x_train_set['horsepower']),list(x_train_set['acceleration']),list(x_train_set['weight']),list(x_train_set['year'])]));
y_train_array = np.array(y_train_set);
x_test_array = np.transpose(np.array([list(x_test_set['cylinders']),list(x_test_set['horsepower']),list(x_test_set['acceleration']),list(x_test_set['weight']),list(x_test_set['year'])]));
y_test_array = np.array(y_test_set);

#part d - Fitting a Linear Discrimiant to Auto Data
def Q5_d(x_train, y_train, x_test, y_test):
    linearDisc = LinearDiscriminantAnalysis();
    linearDisc.fit(x_train, y_train);
    predictions_LDA = linearDisc.predict(x_test);
    Error_LDA = 0.0
    #Find Test error rate
    for i in range(len(predictions_LDA)):
        if predictions_LDA[i] != y_test[i]:
            Error_LDA = Error_LDA + 1.0;
    Error_LDA = Error_LDA/len(predictions_LDA);
    print("Linear discriminant analysis Test Error Rate: ", Error_LDA);



#part e - Fitting a Quadratic Discriminant to Auto Data
def Q5_e(x_train,y_train, x_test, y_test):
    quadDisc = QuadraticDiscriminantAnalysis();
    quadDisc.fit(x_train, y_train);
    predictions_QDA = quadDisc.predict(x_test);
    Error_QDA = 0.0;
    #Find Test error rate
    for i in range(len(predictions_QDA)):
        if predictions_QDA[i] != y_test[i]:
            Error_QDA = Error_QDA + 1.0;
    Error_QDA = Error_QDA/len(predictions_QDA);
    print("Quadratic Discriminant Analysis Test Error Rate: ", Error_QDA);

#part g - Fitting Gaussian Naive Bayes to Auto data
def Q5_g(x_train,y_train, x_test, y_test):
    GNB = GaussianNB();
    GNB.fit(x_train, y_train);
    predictions_GNB = GNB.predict(x_test);
    Error_GNB = 0.0
    #Find Test error rate
    for i in range(len(predictions_GNB)):
        if predictions_GNB[i] != y_test[i]:
            Error_GNB = Error_GNB + 1.0;
    Error_GNB = Error_GNB/len(predictions_GNB);
    print("Naive Bayes Test Error Rate: ", Error_GNB);




#Exercise 6 - Chapter 5, Question 8
#---------------------


def Q6(a_seed):
    #part a - generate random variables
    np.random.seed(1);
    x = np.random.normal(size = 100);
    y = x - 2*(np.transpose(x)*x) + np.random.normal(size = 100);

    #part b - create scatterplot of y vs. x
    plt.scatter(x,y, label = 'Generated points');
    plt.xlabel('x');
    plt.ylabel('y');
    plt.legend();
    plt.show();
    plt.clf();

    #part c/d/e
    #define random seed and deine variables - different seeds chosen when method implemented at bottom
    np.random.seed(a_seed);
    x_squared = np.transpose(x)*x;
    x_cubed = np.transpose(x)*x_squared;
    x_fourth = np.transpose(x)*x_cubed;
    data_Vars = np.array([x, x_squared, x_cubed, x_fourth]);


    #Calculate LOOCV error
    for j in range(len(data_Vars)):
        LOOCV = LeaveOneOut();
        LinReg = LinearRegression();
        data_j = data_Vars[0:j + 1];
        data_j = np.transpose(data_j);
        MSE = 0.0
        for train, test in LOOCV.split(data_j):
            test_X = data_j[test[0],:];
            test_y = y[test[0]];
            train_X = np.delete(data_j, test[0], axis = 0);
            train_y = np.delete(y, test[0]);
            LinReg.fit(train_X, train_y);
            MSE = MSE + (test_y - LinReg.predict(test_X.reshape(1, -1))[0])**2
        MSE = MSE/len(y);
        if j + 1 < 4:
            print("MSE for part", "i"*(j + 1),":", MSE);
        else:
            print("MSE for part iv : ",MSE);


    #part f - Run Regression on each model hypothesized in C
    for j in range(len(data_Vars)):
        data_j = np.transpose(data_Vars[0:j+1]);
        data_j = sm.add_constant(data_j)
        linReg = sm.OLS(y, data_j).fit();
        print(linReg.summary());






#Exercise 7 - Chapter 5, Question 9
#---------------------
def Q7(random_seed):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
    #part a - estimate mean by using sample mean
    print("Sample Mean of MEDV:",data_Bos.mean(numeric_only = True)['MEDV']);
    #part b - estimate standard error by using sample standard error
    print("Standard Error of MEDV:", data_Bos.sem(numeric_only = True)['MEDV']);

    num_data_points = data_Bos.shape[0];
    np.random.seed(random_seed);
    mean_Calc = 0.0;
    indiv_Means = [];
    for b in range(1000):
        currSample = data_Bos['MEDV'].sample(n = num_data_points, replace = True);
        currMean = currSample.mean();
        mean_Calc = mean_Calc + currMean;
        indiv_Means.append(currMean);
    
    #create bootstrapped mean
    mean_Calc = mean_Calc/1000;
    indivMeans = np.array(indiv_Means);
    shiftedMeans = indivMeans - np.array([mean_Calc]*1000);
    shiftedMeans_SQ = np.transpose(shiftedMeans)*shiftedMeans;

    #part c - estimate standard error using bootstrap
    bootStrap_Error = math.sqrt(np.sum(shiftedMeans_SQ)/(1000 - 1));
    print("Bootstrap Mean Calculation:", mean_Calc);
    print("Bootstrap Standard Error Approximation:", bootStrap_Error);

    #part d - create confidence interval tuple
    Conf_Int = (mean_Calc - (2*bootStrap_Error),mean_Calc + (2*bootStrap_Error));
    print("95% Confidence Interval for Mean:", Conf_Int);

    #part e - use sample median as estimate for median
    print("Sample Median of MEDV:",data_Bos.median(numeric_only = True)['MEDV']);


    median_Calc = 0.0;
    indivMedians = [];
    for d in range(1000):
        currSample = data_Bos['MEDV'].sample(n = num_data_points, replace = True);
        currMedian = currSample.median();
        median_Calc = median_Calc + currMedian;
        indivMedians.append(currMedian);
    
    #create bootstrapped sample medoan
    median_Calc = median_Calc/1000;
    indivMedians = np.array(indivMedians);
    shiftedMedians = indivMedians - np.array([median_Calc]*1000);
    shiftedMedians_SQ = np.transpose(shiftedMedians)*shiftedMedians;
    
    #part f - create bootstrapped median error
    bootStrap_Median_Error = math.sqrt(np.sum(shiftedMedians_SQ)/(1000 - 1));
    print("Bootstrap Median Calculation:", median_Calc);
    print("Bootstrap Median Standard Error Approximation:", bootStrap_Median_Error);

    #part g - use sample 10th percentile as estimate for 10th percenitle
    print("Sample 10th Percentile of MEDV:",data_Bos.quantile(.1, numeric_only = True)['MEDV']);

    
    tenth_Calc = 0.0;
    indiv_Tenths = [];
    for t in range(1000):
        currSample = data_Bos['MEDV'].sample(n = num_data_points, replace = True);
        currTenth = currSample.quantile(.1);
        tenth_Calc = tenth_Calc + currTenth;
        indiv_Tenths.append(currTenth);
    
    #create sample 10th percentile
    tenth_Calc = tenth_Calc/1000;
    indiv_Tenths = np.array(indiv_Tenths);
    shiftedTenths = indiv_Tenths - np.array([tenth_Calc]*1000);
    shiftedTenths_SQ = np.transpose(shiftedTenths)*shiftedTenths;

    #part h - create bootstrapped 10th percentile error
    bootStrap_Tenths_Error = math.sqrt(np.sum(shiftedTenths_SQ)/(1000 - 1));
    print("Bootstrap 10th Percentile calculation:", tenth_Calc);
    print("Bootstrap 10th Percentile Standard Error Approximation:", bootStrap_Tenths_Error)



#Exercise 8 - Chapter 7, Question 6
#--------------------
#part a

#a_1: find CV error for all different polynomial degree fits
def Q8_a_1(random_seeding):
    kFold_Error = [];
    data_Wage = pd.read_csv('Wage.csv');
    np.random.seed(random_seeding);
    for d in range(1, 11):
        age_Vals = data_Wage['age'].to_numpy();
        y_vals = data_Wage['wage'].to_numpy();
        X_Vals = [age_Vals];
        c = 1;
        while c < d:
            curr_age_Vals = np.array(X_Vals[len(X_Vals) - 1]);
            next_age_Vals = np.transpose(curr_age_Vals)*age_Vals;
            X_Vals.append(next_age_Vals);
            c = c + 1;
        X_Vals = np.array(X_Vals);
        X_Vals = np.transpose(X_Vals);
        kFold_Validator = KFold(shuffle = True);
        columnHeaders = ["age^{}".format(i) for i in range(1, d + 1)];
        currError = 0.0;
        for i, (train_indices, test_indices) in enumerate(kFold_Validator.split(X_Vals, y = y_vals)):
            x_values = pd.DataFrame(data = X_Vals, columns = columnHeaders);
            y_values = pd.DataFrame(data = y_vals, columns = ['wage']);
            x_values = sm.add_constant(x_values);
            x_train = x_values.iloc[train_indices];
            y_train = y_values.iloc[train_indices];
            x_test = x_values.iloc[test_indices];
            y_test = y_values.iloc[test_indices];
            linReg = sm.OLS(y_train, x_train).fit();
            predictions = linReg.predict(x_test).to_numpy();
            y_test = y_test.to_numpy().flatten();
            currError += MeanSquaredError(y_test, predictions);
        kFold_Error.append(currError/5)
    for degree in range(len(kFold_Error)):
        print("degree {} polynmial Error:".format(degree + 1), kFold_Error[degree]);

#a_2: after seeing which one has the lowest error, plot that polynomial degree fit
def Q8_a_2(deg_polynomial):
    data_Wage = pd.read_csv('Wage.csv');
    ageVals = data_Wage['age'].to_numpy();
    y_vals = data_Wage['wage'].to_numpy();
    X_vals = [ageVals];
    c = 1;
    while c < deg_polynomial:
        curr_age_Vals = np.array(X_vals[len(X_vals) - 1]);
        next_age_Vals = np.transpose(curr_age_Vals)*ageVals;
        X_vals.append(next_age_Vals);
        c = c + 1;
    X_vals = np.array(X_vals);
    X_vals = np.transpose(X_vals);
    columnHeaders = ["age^{}".format(i) for i in range(1, deg_polynomial + 1)];
    x_values = pd.DataFrame(data = X_vals, columns = columnHeaders);
    y_values = pd.DataFrame(data = y_vals, columns = ['wage']);
    x_values = x_values = sm.add_constant(x_values);
    linReg = sm.OLS(y_values, x_values).fit();
    coefficients = np.array(linReg.params);
    x = np.linspace(15, 85, num = 1000);
    f_x = [coefficients[0]]*len(x);
    x_curr = x;
    for coef in range(1,len(coefficients)):
        f_x = f_x + coefficients[coef]*x_curr;
        x_curr = np.transpose(x_curr)*x;
    plt.plot(x,f_x, label = 'OLS fit');
    plt.scatter(ageVals,y_vals, color = 'orange');
    plt.xlabel("Age");
    plt.ylabel("Wage");
    plt.title("Wage as a function of Age with degree {} polynomial fit".format(deg_polynomial));
    plt.legend();
    plt.show();
    plt.clf();

#part b - NOT COMPLETE; creates step functions during kFOLD validation, does not complete CV error to optimize for number of splits
def Q8_b(max_cuts, num_splits, random_seeding):
    data_Wage = pd.read_csv("Wage.csv");
    X_Vals = data_Wage['age'];
    Y_Vals = data_Wage['wage'];
    np.random.seed(random_seeding);
    MSE_calculator = [];
    for i in range(1, max_cuts):
        kFold_Validator = KFold(n_splits = num_splits, shuffle = True);
        curr_MSE = 0.0;
        for j, (train_indices, test_indices) in enumerate(kFold_Validator.split(X_Vals, Y_Vals)):
                x_train = X_Vals.iloc[train_indices];
                y_train = Y_Vals.iloc[train_indices];
                x_test = X_Vals.iloc[test_indices];
                y_test = Y_Vals.iloc[test_indices];
                dw_cut, bins = pd.cut(x_train,i + 1, retbins = True);
                dw_steps = pd.concat([x_train, dw_cut, y_train], keys = ['age', 'age cuts', 'wage'], axis = 1);
                dw_dummies = pd.get_dummies(dw_steps['age cuts']);
                dw_dummies = sm.add_constant(dw_dummies);
                dw_dummies = dw_dummies.drop(dw_dummies.columns[1], axis = 1);
                regressor = sm.GLM(dw_steps.wage, dw_dummies).fit();
                #print(regressor.summary());
                bin_mapping = np.digitize(x_test.ravel(), bins);
                x_test = sm.add_constant(pd.get_dummies(bin_mapping).drop(1, axis = 1));
    plt.scatter(data_Wage['age'], data_Wage['wage']);
    plt.show();
    plt.clf();


#Exercise 9 - Chapter 7, Question 9
#--------------------
#part a - fitting a cubic polynomial using housing.csv
def Q9_a():
    cubic_regressor = Q9_helper(3);
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
    x = np.linspace(0, 13, num = 1000);
    coefficients = np.array(cubic_regressor.params);
    x_sq = np.transpose(x)*x;
    x_cb = np.transpose(x_sq)*x;
    f_x = coefficients[0] + coefficients[1]*x + coefficients[2]*x_sq + coefficients[3]*x_cb;
    plt.plot(x, f_x, label = 'OLS fit', color = 'red');
    plt.scatter(data_Bos['DIS'], data_Bos['NOX']);
    plt.xlabel('DIS(Weighted distance to 5 Employment centers)');
    plt.ylabel('NOX(ppm)')
    plt.title('NOX compared to DIS, with cubic polynomial fit');
    plt.legend();
    print(cubic_regressor.summary());
    plt.show();
    plt.clf();

#part b - fit polynomials to data from 1-10 degree
def Q9_b(max_degree):
    np.random.seed(100);
    RSS_array = [];
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    plt.scatter(data_Bos['DIS'], data_Bos['NOX']);
    plt.xlabel('DIS(Weighted distance to 5 Employment centers)');
    plt.ylabel('NOX(ppm)')
    plt.title('NOX compared to DIS, with polynomial fits');
    for i in range(1, max_degree + 1):
        regressor = Q9_helper(i);
        x = np.linspace(1, 11, num = 1000);
        coefficients = np.array(regressor.params);
        f_x = len(x)*[coefficients[0]];
        curr_x = x
        for j in range(1, len(coefficients)):
            f_x = f_x + curr_x*coefficients[j];
            curr_x = np.transpose(curr_x)*x;
        color_array = np.random.random(size = 3);
        color_array = list(color_array)
        plt.plot(x, f_x, color = color_array, label = 'degree {}'.format(i));
        RSS_array.append(regressor.ssr);
        print("Degree {} polynomial SSR:".format(i), regressor.ssr);
    plt.legend();
    plt.show();
    plt.clf();


#part c - same as b, but using cross validation to try and find the optimal degree of polynomial fit
def Q9_c(max_degree):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    dis_vals = data_Bos['DIS'].to_numpy();
    y_Vals = data_Bos['NOX'].to_numpy();
    MSE_array = [];
    for i in range(1, max_degree + 1):
        X_Vals = [dis_vals];
        for j in range(1,i):
            curr_dis_vals = np.array(X_Vals[len(X_Vals) - 1]);
            next_dis_vals = np.transpose(curr_dis_vals)*dis_vals;
            X_Vals.append(next_dis_vals);
        X_Vals = np.array(X_Vals);
        X_Vals = np.transpose(X_Vals);
        column_Headers = ["dis^{}".format(f) for f in range(1, i + 1)];
        x_values = pd.DataFrame(data = X_Vals, columns = column_Headers);
        y_values = pd.DataFrame(data = y_Vals, columns = ['nox']);
        x_values = sm.add_constant(x_values);
        curr_MSE = 0.0
        LOOCV = LeaveOneOut();
        for l, (train_indices, test_index) in enumerate(LOOCV.split(x_values,y_values)):
            x_train = x_values.iloc[train_indices];
            y_train = y_values.iloc[train_indices];
            x_test = x_values.iloc[test_index];
            y_test = float(y_values.iloc[test_index]['nox']);
            linReg = sm.OLS(y_train, x_train).fit();
            curr_MSE = curr_MSE + (y_test - float(linReg.predict(x_test)))**2;
        curr_MSE = curr_MSE/y_Vals.shape[0];
        print("Degree {} polynomial LOOCV error:".format(i), curr_MSE);
        MSE_array.append(curr_MSE);
    MSE_array = np.array(MSE_array);
    min_Degree = np.argmin(MSE_array) + 1;
    print("Degree of polynomial with minimum LOOCV error:", min_Degree);

#part e - spline regression using degree 3 polynomial
def Q9_e(deg_polynomial, max_knots):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    plt.scatter(data_Bos['DIS'],data_Bos['NOX']);
    RSS_array = [];
    np.random.seed(50);
    for i in range(max_knots + 1):
        model, RSS, x_vals, y_pred = Q9_helper_2(deg_polynomial, i);
        #Reporting RSS of the regression with i splines
        print("RSS of {} degree spline with {} knots ({} degrees of freedom):".format(deg_polynomial, i, i + deg_polynomial + 1), RSS);
        plt.plot(x_vals, y_pred, color = np.random.random(size = 3), label = '{} knots'.format(i));
    plt.title('{} degree polynomial regression splines with knots, NOX vs. DIS'.format(deg_polynomial));
    plt.xlabel('DIS(Weighted distance to 5 Employment centers)');
    plt.ylabel('NOX(ppm)');
    plt.legend();
    plt.show();
    plt.clf();

#part f - spline regression again, just with cross validation to determine best number of notes
def Q9_f(deg_polynomial, max_knots, random_seed, num_splits):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    X_vals = data_Bos['DIS'];
    y_vals = data_Bos['NOX'];
    np.random.seed(random_seed);
    list_Errors = [];
    for i in range(max_knots + 1):
        knot_Quantiles = [j/(i + 1) for j in range(1, i + 1)];
        knot_Vals = [X_vals.quantile(q) for q in knot_Quantiles];
        tup_Locations = tuple(knot_Vals)
        kFold_Validator = KFold(n_splits = num_splits, shuffle = True);
        Mean_Squared_Error = 0.0;
        for p, (train_indices, test_indices) in enumerate(kFold_Validator.split(X_vals, y_vals)):
            x_train = X_vals.iloc[train_indices];
            y_train = y_vals.iloc[train_indices];
            x_test = X_vals.iloc[test_indices];
            y_test = y_vals.iloc[test_indices];
            y_test = y_test.to_numpy().flatten();
            basis_x = dmatrix("bs(x_train, knots = tup_Locations, degree = deg_polynomial, include_intercept = False)", {"x_train": x_train},return_type = 'dataframe');
            regressor = sm.GLM(y_train, basis_x).fit();
            pred = regressor.predict(dmatrix("bs(x_test, knots = tup_Locations, degree = deg_polynomial, include_intercept = False)", {"x_test": x_test},return_type = 'dataframe'));
            pred = pred.to_numpy().flatten();
            Mean_Squared_Error = Mean_Squared_Error + MeanSquaredError(pred,y_test);
        #CV error with i splines
        Mean_Squared_Error = Mean_Squared_Error/num_splits;
        list_Errors.append(Mean_Squared_Error);
        print("Average {}-Fold MSE, {} knots:".format(num_splits, i), Mean_Squared_Error)

#Q5_d(x_train_array,y_train_array,x_test_array,y_test_array);
#Q5_e(x_train_array,y_train_array,x_test_array,y_test_array);
#Q5_g(x_train_array,y_train_array,x_test_array,y_test_array);
#Q6(3);
#Q7(4);
#Q8_a_1(5);
#Q8_a_2(deg_polynomial = 4);
#Q8_b(max_cuts = 6, num_splits = 5, random_seeding = 105);
#Q9_a();
#Q9_b(max_degree = 10);
#Q9_c(max_degree = 10)
#Q9_e(deg_polynomial = 3,, max_knots = 8)
#Q9_f(deg_polynomial = 3, max_knots = 8, random_seed = 35, num_splits = 5)
