
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import math;
from sklearn.model_selection import train_test_split;
from sklearn import tree, svm;
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier;
from sklearn.datasets import make_blobs;
from sklearn import model_selection;




#Helper Methods
#--------------
def get_Housing_data():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_Bos = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names);
    return data_Bos;

def MeanSquaredError(y_vals, y_hat):
    diff = y_vals - y_hat;
    diff_sq = np.transpose(diff)*diff
    mse = np.sum(diff_sq)/len(y_vals);
    return mse;

def Error_Rate(y_vals, y_hat):
    error_arr = np.array([int(y_vals[i] != y_hat[i]) for i in range(len(y_vals))]);
    error_Rate = np.sum(error_arr)/len(y_vals);
    return error_Rate;

def add_CRIM_dummy():
    data_Bos = get_Housing_data();
    bos_New = data_Bos.assign(CRIM01 = (data_Bos.CRIM >= data_Bos.median(numeric_only = True)['CRIM']));
    bos_New['CRIM01'] = bos_New['CRIM01'].replace({True: 1, False: 0});
    return bos_New;

def Covariance_Q7(y, y_hat, n):
    cov_matrix = [];
    for i in range(len(y)):
        y_i_centered = y[i] - np.average(y[i]);
        y_hat_i_centered = y_hat[i] - np.average(y_hat[i]);
        varying_val = np.matmul(y_i_centered.T, y_hat_i_centered);
        cov_matrix.append(varying_val/(n - 1));
    cov_vector = np.array(cov_matrix);
    return cov_vector;



#Programming Questions
#--------------------

#Question 6 - Extension of Secion 8.4, Exercise 7

def Q6_a(random_seed = 0):
    #Get DataSet
    data_Bos = get_Housing_data();
    y = data_Bos.pop('MEDV');
    #Get training, testing data, create decision Tree
    X_train, X_test, y_train, y_test = train_test_split(data_Bos, y, test_size = .5, random_state = random_seed);
    regTree = tree.DecisionTreeRegressor(random_state = random_seed);
    #Do cost_complexity pruning to generate effective alphas
    path = regTree.cost_complexity_pruning_path(X_train, y_train);
    effec_alphas, impurities = path.ccp_alphas, path.impurities;
    #Remove maximum alpha - this creates a trivial, root node
    effec_alphas = effec_alphas[:-1];
    MSE = [];
    #Generate MSE's
    for a in effec_alphas:
        test_Tree = tree.DecisionTreeRegressor(random_state = random_seed, ccp_alpha = a);
        test_Tree.fit(X_train, y_train);
        y_hat = test_Tree.predict(X_test);
        MSE.append(MeanSquaredError(y_test, y_hat));
    #Find tree with minimum mean squared Error
    MSE = np.array(MSE);
    arg_min, min_MSE = np.argmin(MSE), np.min(MSE)
    min_MSE_alpha = effec_alphas[arg_min];

    #Fit, and then plot tree with best MSE
    best_Tree = tree.DecisionTreeRegressor(random_state = random_seed, ccp_alpha = min_MSE_alpha);
    best_Tree.fit(X_train, y_train)
    tree.plot_tree(best_Tree, feature_names = list(data_Bos.columns), impurity = False, fontsize = 8);
    plt.show();
    plt.clf();
    print("Optimal Alpha: ", min_MSE_alpha);
    print("MSE of Optimal Tree: ", min_MSE);


def Q6_b(random_seed = 1):
    #get dataset
    data_Bos = get_Housing_data();
    y = data_Bos.pop('MEDV');
    #Get training, testing data, create decision Tree
    X_train, X_test, y_train, y_test = train_test_split(data_Bos, y, test_size = .5, random_state = random_seed);
    #create the estimator list for different numbers of estimators
    n_est_list = [25, 100];
    MSE = []
    #create and get results for forests
    for n_est in n_est_list:
        twoFiveForest = RandomForestRegressor(n_estimators = n_est, min_samples_leaf = 5, max_features = 6, random_state = random_seed);
        twoFiveForest.fit(X_train, y_train);
        y_hat = twoFiveForest.predict(X_test);
        MSE.append(MeanSquaredError(y_test, y_hat));
        #Collect importance of each individual features:
        feature_importances = np.vstack((list(data_Bos.columns), twoFiveForest.feature_importances_)).T;
        feature_importances_df = pd.DataFrame(data = feature_importances, columns = ['feature', 'feature_importance']);
        feature_importances_df['feature'] = feature_importances_df['feature'].map(str);
        feature_importances_df['feature_importance'] = feature_importances_df['feature_importance'].map(float);
        feature_importances_df_sorted = feature_importances_df.sort_values('feature_importance', ascending = False);
        #plot individual feature importances
        plt.barh(feature_importances_df_sorted['feature'], feature_importances_df_sorted['feature_importance']);
        plt.xlabel('Feature Importance', size = 15);
        plt.title("Feature Importances with {} trees".format(n_est), size = 15);
        plt.show();
        plt.clf();
    print("MSE with 25 trees: ", MSE[0]);
    print("MSE with 100 trees: ", MSE[1]);



def Q6_c(random_seed = 2):
    #Set up Data Set
    bos_New = add_CRIM_dummy();
    bos_New.pop('CRIM');
    y = bos_New.pop('CRIM01');
    X_train, X_test, y_train, y_test = train_test_split(bos_New, y, test_size = .5, random_state = random_seed);
    classTree = tree.DecisionTreeClassifier(random_state = random_seed, min_samples_leaf = 3);
    #Get effective alphas that change tree itself
    path = classTree.cost_complexity_pruning_path(X_train, y_train);
    effec_alphas, impurities = path.ccp_alphas, path.impurities;
    #Remove biggest alpha that corresponds to tree that is just root linear_model
    effec_alphas = effec_alphas[:-1];
    #Generate Error rates for each different alpha
    error_rates = [];
    y_test = y_test.to_numpy();
    for a in effec_alphas:
        new_Tree = tree.DecisionTreeClassifier(random_state = random_seed, min_samples_leaf = 3, ccp_alpha = a);
        new_Tree.fit(X_train, y_train);
        y_hat = new_Tree.predict(X_test);
        error_rates.append(Error_Rate(y_test, y_hat));

    #Find (simplest) tree with minimum Error rate
    error_rates = np.array(error_rates);
    arg_min, min_error = np.where(error_rates == error_rates.min()), np.min(error_rates);
    min_error_alpha = effec_alphas[arg_min[-1][0]];

    #Fit, and then plot tree with best MSE
    best_Tree = tree.DecisionTreeClassifier(random_state = random_seed, min_samples_leaf = 3, ccp_alpha = min_error_alpha);
    best_Tree.fit(X_train, y_train)
    tree.plot_tree(best_Tree, feature_names = list(bos_New.columns), class_names = ["Below", "Above"], proportion = True, fontsize = 9);
    plt.show();
    plt.clf();
    print("Optimal Alpha: ", min_error_alpha);
    print("Error Rate of Optimal Tree: ", min_error);



def Q6_d(random_seed = 3):
    #Set up Data Set
    bos_New = add_CRIM_dummy();
    bos_New.pop('CRIM');
    y = bos_New.pop('CRIM01');
    X_train, X_test, y_train, y_test = train_test_split(bos_New, y, test_size = .5, random_state = random_seed);
    #create the estimator list for different numbers of estimators
    n_est_list = [25, 100];
    error_rates = [];
    y_test = y_test.to_numpy();
    #create and get results for forests
    for n_est in n_est_list:
        randForest = RandomForestClassifier(n_estimators = n_est, min_samples_leaf = 5, max_features = 6, random_state = random_seed);
        randForest.fit(X_train, y_train);
        y_hat = randForest.predict(X_test);
        error_rates.append(Error_Rate(y_hat, y_test));
        #Collect Importance of Individual Features
        feature_importances = np.vstack((list(bos_New.columns), randForest.feature_importances_)).T;
        feature_importances_df = pd.DataFrame(data = feature_importances, columns = ['feature', 'feature_importance']);
        feature_importances_df['feature'] = feature_importances_df['feature'].map(str);
        feature_importances_df['feature_importance'] = feature_importances_df['feature_importance'].map(float);
        feature_importances_df_sorted = feature_importances_df.sort_values('feature_importance', ascending = False);
        #plot feature importances
        plt.barh(feature_importances_df_sorted['feature'], feature_importances_df_sorted['feature_importance']);
        plt.xlabel('Feature Importance', size = 15);
        plt.title("Feature Importances with {} trees".format(n_est), size = 15);
        plt.show();
        plt.clf();
    print("Error Rate with 25 trees: ", error_rates[0]);
    print("Error Rate with 100 trees: ", error_rates[1]);


#Question 7 - Chapter 9, Exercise 5 (From Elements of Statistical Learning)
def Q7(n, random_seed = 4):
    #part b
    np.random.seed(random_seed);
    x_1 = np.random.normal(size = 100);
    df = pd.DataFrame(data = x_1, columns = ['X_1']);
    for i in range(2,n + 1):
        x = np.random.normal(size = 100);
        df['X_{}'.format(i)] = x;
    degree_freedom_est = [0.0, 0.0, 0.0];
    y_vals = np.empty(shape = (100,1));
    #generate empty vectors for 1, 5, 10 leaf node predictions
    y_hats_1 = np.empty(shape = (100,1));
    y_hats_5 = np.empty(shape = (100,1));
    y_hats_10 = np.empty(shape = (100,1));
    #part c
    for j in range(n):
        y = np.random.normal(size = 100);
        #create y matrix
        if j == 0:
            y_vals = y.reshape((100,1))
        else:
            y_vals = np.append(y_vals, y.reshape((100,1)), axis = 1)
        #Generate 1 leaf tree
        base_Tree = tree.DecisionTreeRegressor(random_state = random_seed);
        path = base_Tree.cost_complexity_pruning_path(df, y);
        effec_alphas = path.ccp_alphas;
        alpha_base = effec_alphas[-1];
        base_Tree_2 = tree.DecisionTreeRegressor(random_state = random_seed, ccp_alpha = alpha_base);
        base_Tree_2.fit(df, y);
        if j == 0:
            y_hats_1 = base_Tree_2.predict(df).reshape((100,1));
        else:
            y_hats_1 = np.append(y_hats_1, base_Tree_2.predict(df).reshape((100,1)), axis = 1);
        n_nodes = [5,10];
        #Generate 5 and 10 leaf trees
        for k in n_nodes:
            regTree = tree.DecisionTreeRegressor(random_state = random_seed, max_leaf_nodes = k);
            regTree.fit(df, y);
            if k == 5:
                if j == 0:
                    y_hats_5 = regTree.predict(df).reshape((100,1));
                else:
                    y_hats_5 = np.append(y_hats_5, regTree.predict(df).reshape((100,1)), axis = 1);
            elif k == 10:
                if j == 0:
                    y_hats_10 = regTree.predict(df).reshape((100,1));
                else:
                    y_hats_10 = np.append(y_hats_10, regTree.predict(df).reshape((100,1)), axis = 1);
    Cov_Vectors = [Covariance_Q7(y_vals, y_hats_1, n), Covariance_Q7(y_vals, y_hats_5, n), Covariance_Q7(y_vals, y_hats_10, n)];
    deg_freedom_vector = np.array([np.sum(i) for i in Cov_Vectors]);
    print("Degrees of Freedom, 1 Leaf: ", deg_freedom_vector[0]);
    print("Degrees of Freedom, 5 Leaves: ", deg_freedom_vector[1]);
    print("Degrees of Freedom, 10 Leaves: ", deg_freedom_vector[2]);


#Question 8 - Secion 9.7, Question 6
def Q8(random_seed = 5):
    #Part a
    X, y_vals = make_blobs(n_samples = 200, centers = np.array([(1,1), (4,5)]), random_state = random_seed);
    clf = svm.SVC(kernel = "linear", C = 1000000000)
    clf.fit(X,y_vals);
    y_hat = clf.predict(X);
    print("Error rate with (effectively) no regularization: ", Error_Rate(y_hat, y_vals));

    plt.scatter(X[:, 0], X[:, 1], c=y_vals, s=30, cmap=plt.cm.Paired);
    plt.xlabel("X_1");
    plt.ylabel("X_2");
    plt.title("2 class, 2 feature, linearly seperable data")
    plt.show();
    plt.clf();

    #Part B
    c_vector = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.009];
    c_train_errors = [];
    c_test_errors = [];
    #part C stuff
    part_c_test_errors = [];
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_vals, test_size = .5, random_state = random_seed);

    for c in c_vector:
        kFold_Validator = model_selection.KFold(shuffle = True, random_state = random_seed);
        curr_error_train = 0.0
        curr_error_test = 0.0
        #Part B
        for i, (train_indices, test_indices) in enumerate(kFold_Validator.split(X, y = y_vals)):
            x_values = pd.DataFrame(data = X);
            y_values = pd.DataFrame(data = y_vals);
            x_train = x_values.iloc[train_indices];
            y_train = y_values.iloc[train_indices];
            x_test = x_values.iloc[test_indices];
            y_test = y_values.iloc[test_indices];
            svClassifier = svm.SVC(kernel = 'linear', C = c);
            svClassifier.fit(x_train, y_train.to_numpy().flatten());
            predictions_train = svClassifier.predict(x_train);
            predictions_test = svClassifier.predict(x_test);
            curr_error_train = curr_error_train + Error_Rate(predictions_train, y_train.to_numpy().flatten());
            curr_error_test = curr_error_test + Error_Rate(predictions_test, y_test.to_numpy().flatten());
        c_train_errors.append(round(curr_error_train/5, 5));
        c_test_errors.append(curr_error_test/5);

        #part C
        spec_Classifier = svm.SVC(kernel = 'linear', C = c);
        spec_Classifier.fit(X_train, Y_train);
        predictions_c = spec_Classifier.predict(X_test);
        part_c_test_errors.append(Error_Rate(predictions_c, Y_test));
    #printing part B results
    print("Cost parameters: ", c_vector)
    print("CV Train Errors: ", c_train_errors);
    print("CV Test Errors: ", c_test_errors);

    #printing part C results
    print("Testing Error: ", part_c_test_errors)

#Q6_a();
#Q6_b();
#Q6_c();
#Q6_d();
#Q7(n = 10);
#Q8();
