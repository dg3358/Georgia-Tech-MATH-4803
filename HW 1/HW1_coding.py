import pandas as pd;
import numpy as np;
import math;
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

#QUESTION 1
#---------

#Read in Data
df = pd.read_csv('Auto.data', sep='\s+')


#Find correct indices (ones with vals for horsepower and mpg)
lengthData = len(df['mpg'])
list_Indices = np.array([i for i in range(lengthData)])
excludedRows = df.loc[df['horsepower'] == '?'].index.values
regressionVals = np.setdiff1d(list_Indices, excludedRows);

#create new dataframe including only non-? horsepower rows
df_new = df.loc[regressionVals];
df_new["horsepower"] = pd.to_numeric(df_new["horsepower"])
#print(df_new[['mpg','horsepower']].describe())

#Run Regression
result = sm.ols(formula = "mpg ~ horsepower", data = df_new).fit();
interceptVal = result.params["Intercept"];
betaVal = result.params["horsepower"]
#print(result.summary());

#plot
plt.scatter(df_new["horsepower"], df_new["mpg"], label = 'datapoints');
axes = plt.gca();
x_vals = np.array(axes.get_xlim());
y_vals = interceptVal + betaVal * x_vals;
plt.plot(x_vals, y_vals, color = 'black', label = 'LSR line')
plt.xlabel('Horsepower');
plt.ylabel('mpg');
plt.legend();
plt.show();
plt.clf();

#Print SSR
#print("Sum of squared Residuals: ", result.ssr)






#QUESTION 2
#--------

#Print correlation matrix
corrMat = df_new.corr();
#print(corrMat);

#run multiple linear regressionVals
result2 = sm.ols(formula = "mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin", data = df_new).fit();
#print(result2.summary());

#RSS part # TODO:
#print("Sum of squared Residuals: ", result2.ssr);







#QUESTION 3
#---------
np.random.seed(1);

#part a
xvars = np.random.normal(0,1,100);
#print(xvars)

#part b
eps = np.random.normal(0,.5,100);
#print(eps);

#part c
y = -1 + .5*xvars + eps;
#print("Y: ",y);

#part d
plt.scatter(xvars, y);
plt.xlabel("X-vals");
plt.ylabel("Y-vals");
#plt.show();

#part e
result3 = sm.ols(formula = "y ~ xvars", data = {"xvars": xvars, "y": y}).fit();
#print(result3.summary());

#part f
axes = plt.gca();
x_valsTrue = np.array(axes.get_xlim());
y_valsTrue = -1 + .5*x_valsTrue;
plt.plot(x_valsTrue, y_valsTrue, color = 'black', label = 'True line');
interceptValActual = result3.params["Intercept"];
betaValActual = result3.params["xvars"];
y_valsActual = interceptValActual + x_valsTrue*betaValActual;
plt.plot(x_valsTrue, y_valsActual, color = 'red', label = 'Regression Line');
plt.title("Regression: Variance = .25")
plt.legend();
plt.show();
plt.clf();

#part g
xvarssq = np.array([xvars[i]**2 for i in range(len(xvars))]);
result4 = sm.ols(formula = "y ~ xvars + xvarssq", data = {"xvars": xvars, "y": y, "xvarssq": xvarssq}).fit();
#print(result4.summary());

#part h - have to redefine xvars, eps, y
xvars = np.random.normal(0,1,100);
eps = np.random.normal(0,.2, 100);
y = -1 + .5*xvars + eps;
plt.scatter(xvars, y);
result5 = sm.ols(formula = "y ~ xvars", data = {"xvars": xvars, "y": y}).fit();
interceptValActual = result5.params["Intercept"];
betaValActual = result5.params["xvars"];
y_valsActual = interceptValActual + x_valsTrue*betaValActual;
#print(result5.summary());
plt.plot(x_valsTrue, y_valsTrue, color = 'black', label = 'True line');
plt.plot(x_valsTrue, y_valsActual, color = 'red', label = 'Regression Line');
plt.title(label = "Regression: Variance = .04");
plt.legend();
plt.show();
plt.clf();


#part i - have to redefine xvars, eps, y
xvars = np.random.normal(0,1,100);
eps = np.random.normal(0,1,100);
y = -1 + .5*xvars + eps;
plt.scatter(xvars, y);
result6 = sm.ols(formula = "y ~ xvars", data = {"xvars": xvars, "y": y}).fit();
interceptValActual = result6.params["Intercept"];
betaValActual = result6.params["xvars"];
y_valsActual = interceptValActual + x_valsTrue*betaValActual;
#print(result6.summary());
plt.plot(x_valsTrue, y_valsTrue, color = 'black', label = 'True line');
plt.plot(x_valsTrue, y_valsActual, color = 'red', label = 'Regression Line');
plt.legend();
plt.title("Regression: Variance = 1");
plt.show();
plt.clf();
