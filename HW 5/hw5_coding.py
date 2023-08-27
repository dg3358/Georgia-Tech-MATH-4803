import numpy as np;
from tensorflow import keras;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import KFold;
from sklearn.linear_model import LogisticRegression;


#helper Functions

def Euclidean_Norm(x, y):
    return np.linalg.norm(x - y);

def Centroid_Computer(Centroids, clusters, colorings, observations):
    for coloring in colorings:
        curr_centroid = np.array([0,0]);
        num_curr_color = 0
        for i in range(len(clusters)):
            if clusters[i] == coloring:
                curr_centroid[0] = curr_centroid[0] + observations[i][0];
                curr_centroid[1] = curr_centroid[1] + observations[i][1];
                num_curr_color += 1;
        curr_centroid = curr_centroid/num_curr_color;
        Centroids[coloring] = curr_centroid;
    return Centroids;

def Error_Rate(y_vals, y_hat):
    error_arr = np.array([int(y_vals[i] != y_hat[i]) for i in range(len(y_vals))]);
    error_Rate = np.sum(error_arr)/len(y_vals);
    return error_Rate;

def Q5_cd_helper(beta_0, learning_rate):
    curr_val = beta_0;
    beta_vals = [beta_0];
    next_val = beta_0 - learning_rate * (np.cos(beta_0) + .1);
    while np.sin(next_val) + next_val/10 < np.sin(curr_val) + curr_val/10:
        beta_vals.append(next_val);
        curr_val = next_val;
        next_val = next_val - learning_rate * (np.cos(next_val) + .1);
    return beta_vals;



#Question 3
def Q3():
    X_1 = np.array([1,1, 0, 5, 6, 4]);
    X_2 = np.array([4,3,4,1,2,0]);
    plt.scatter(X_1, X_2, label = 'observations', );
    plt.title('Plot of Observations');
    plt.xlabel("X_1");
    plt.ylabel("X_2");
    plt.legend();
    plt.show();
    plt.clf();
    color_labels = np.array(['red', 'blue']);

    #Starting the algorithm
    np.random.seed(10);
    clusters_init = np.random.choice(color_labels, size = len(X_1));
    print(clusters_init)
    observations = np.stack([X_1, X_2], axis = 1);

    #Computer first centroids
    Centroids = {"red": np.array([0,0]) , "blue": np.array([0, 0])};
    curr_centroids = Centroid_Computer(Centroids, clusters_init, color_labels, observations);
    curr_clusters = clusters_init;
    print("Original Centroids:", Centroids);

    iterator_count = 1;
    #run K-Means
    while 2 > 1:
        next_clusters = [];
        for i in range(len(observations)):
            distances_i = np.array([Euclidean_Norm(observations[i], curr_centroids['red']), Euclidean_Norm(observations[i], curr_centroids['blue'])]);
            next_clusters.append(color_labels[np.argmin(distances_i)]);
        curr_centroids = Centroid_Computer(curr_centroids, next_clusters, color_labels, observations);
        if np.array_equal(curr_clusters, next_clusters):
            break;
        if iterator_count == 1:
            print("Clusters after 1 iteration: ", next_clusters);
        iterator_count = iterator_count + 1;
        curr_clusters = next_clusters;
    print("Final Clusters: ", curr_clusters);
    plt.scatter(X_1, X_2, color = curr_clusters);
    plt.title("Data Clustering result from K-Means");
    plt.xlabel("X_1");
    plt.ylabel("X_2");
    plt.show();
    plt.clf();

#Question 5
def Q5_a(start_domain, end_domain):
    x = np.linspace(start_domain, end_domain);
    plt.plot(x, np.sin(x) + x/10);
    plt.xlabel('x');
    plt.ylabel('g(x)');
    plt.title('Function Plot')
    plt.show();



def Q5_cd(beta_0, learning_rate):
    betas = Q5_cd_helper(beta_0, learning_rate);
    print("Minima reached: ", betas[-1]);
    g_x = np.sin(betas) + np.array(betas)/10;
    plt.scatter(betas, g_x, color = 'red', label = 'Gradient Descent Betas');
    plt.legend();
    Q5_a(-6,6);
    plt.clf();


#Question 6
def Q6():
    default = pd.read_csv('Default.csv');
    default['default'] = default['default'].replace({"No": 0, "Yes": 1});
    default['student'] = default['student'].replace({"No": 0, "Yes": 1});
    kFold_Validator = KFold(shuffle = True, random_state = 200);
    X_vals = default;
    y_vals = X_vals.pop("default");
    CV_error_neural_net = [];
    CV_error_logistic = [];
    keras.utils.set_random_seed(2);
    for i, (train_indices, test_indices) in enumerate(kFold_Validator.split(X_vals, y = y_vals)):
        x_train = X_vals.iloc[train_indices];
        x_train = x_train.to_numpy();
        y_train = y_vals.iloc[train_indices];
        y_train = y_train.to_numpy();
        x_test = X_vals.iloc[test_indices];
        x_test = x_test.to_numpy();
        y_test = y_vals.iloc[test_indices];
        y_test = y_test.to_numpy();

        #Neural Network Model
        input_Shape = (x_train.shape[1],);
        inputs = keras.Input(shape = input_Shape, name = "Input_Layer");
        #reshaped_inputs = keras.layers.Reshape(target_shape = (x_train.shape[1], 1), name = "Reshaped_Inputs")(inputs);
        dense_Layer = keras.layers.Dense(10, activation = "relu", name = "Dense_Layer")(inputs);
        dropout_Layer = keras.layers.Dropout(rate = 0.5, name = "Dropout_Layer")(dense_Layer);
        output_layer = keras.layers.Dense(2, activation = "softmax")(dropout_Layer);
        model = keras.Model(inputs = inputs, outputs = output_layer)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4 * 2), loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy']);
        model.summary();
        history = model.fit(x_train, y_train, epochs = 10);
        prediction_probs = model.predict(x_test);
        predictions = np.argmax(prediction_probs, axis = 1);
        CV_error_neural_net.append(Error_Rate(predictions, y_test));

        #Logistic Regression Model
        logitModel = LogisticRegression();
        logitModel.fit(x_train, y_train);
        predictions = logitModel.predict(x_test);
        CV_error_logistic.append(Error_Rate(predictions, y_test));

    CV_error_NN = np.average(CV_error_neural_net);
    CV_error_Logit = np.average(CV_error_logistic);
    print("Neural Net Cross-Validation Error (k = 5): ", CV_error_NN);
    print("Logistic Regression Cross-Validation Error (k = 5): ", CV_error_Logit);


#Calling Functions

Q3();
Q5_a(-6,6);
plt.clf();
Q5_cd(2.3, .1);
Q5_cd(1.4, .1);
Q6();
