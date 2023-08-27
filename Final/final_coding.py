import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn import svm;
from sklearn.inspection import DecisionBoundaryDisplay;
from sklearn.decomposition import PCA;

#Helper Functions
def create_data_set(feature_1_vals, feature_2_vals):
    return np.stack((feature_1_vals, feature_2_vals), axis = 0);

def classify_feature_space(black_Z = True):
    x_02 = np.linspace(0,2, 100);
    red_green_02 = 6 - 2*x_02;
    plt.fill_between(x_02, red_green_02, -4, color = 'green', linewidth = 3, alpha = 0.25);
    plt.fill_between(x_02, red_green_02, 6, color = 'red', linewidth = 3, alpha = 0.25);

    x_225 = np.linspace(2,2.5, 100);
    red_blue_225 = -1.5 + x_225;
    red_green_225 = 6 - 2*x_225;
    plt.fill_between(x_225, red_blue_225, -4, color = 'blue', linewidth = 3, alpha = 0.25);
    plt.fill_between(x_225, red_green_225, 6, color = 'red', linewidth = 3, alpha = 0.25);
    if black_Z == True:
        plt.fill_between(x_225, red_blue_225, red_green_225, color = 'black', linewidth = 3, alpha = 0.5, label = 'Region Z');

    x_256 = np.linspace(2.5, 6, 100);
    red_blue_256 = -1.5 + x_256;
    plt.fill_between(x_256, red_blue_256, -4, color = 'blue', linewidth = 3, alpha = 0.25);
    plt.fill_between(x_256, red_blue_256, 6, color = 'red', linewidth = 3, alpha = 0.25);



#Question 3
def Q3(show_a, show_b, show_c_1, show_c_2, show_de):
    #Part a
    X_1 = np.array([3,2,4,5,3,3,5,5,1,1,0]);
    X_2 = np.array([2,4,4,4,0,1,1,3,1,2,4]);
    color_labels = np.array(['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'green', 'green', 'green']);
    plt.scatter(X_1, X_2, s = 75, color = color_labels);
    plt.title("Plot of Observations with colors");
    plt.xlabel('X_1');
    plt.ylabel('X_2');
    if show_a == True:
        plt.show();
    plt.clf();

    #Part b
    x = np.linspace(0, 5, 100);
    red_green = 6 - 2*x
    blue_green = np.array([2] * len(x));
    red_blue = -1.5 + x
    plt.scatter(X_1, X_2, s = 75, color = color_labels);
    plt.plot(x, red_green, color = 'yellow', label = 'Red-Green Classifier');
    plt.plot(blue_green, np.linspace(-4, 6, 100), color = 'lightseagreen', label = 'Blue-Green Classifier')
    plt.plot(x, red_blue, color = 'purple', label = 'Red-Blue Classifier');
    plt.title("Plot of Observations with Optimal Separating Hyperplanes");
    plt.xlabel('X_1');
    plt.ylabel('X_2');
    plt.legend();
    if show_b == True:
        plt.show();
    plt.clf();

    #Part C
    plt.fill_between(np.linspace(0,5,100), red_blue, -4, color = 'blue', linewidth = 3, alpha = 0.25);
    plt.fill_between(np.linspace(0,5,100), red_blue, 6, color = 'red', linewidth = 3, alpha = 0.25);
    plt.fill_betweenx(np.linspace(-4, 6, 100), 2, 5, color = 'blue', linewidth = 3, alpha= 0.25);
    plt.fill_betweenx(np.linspace(-4, 6, 100), 0, 2, color = 'green', linewidth = 3, alpha= 0.25);
    plt.fill_between(x, red_green, 6, color = 'red', linewidth = 3, alpha = 0.25);
    plt.fill_between(x, red_green, -4, color = 'green', linewidth = 3, alpha = 0.25);
    plt.scatter(X_1, X_2, s = 75, color = color_labels);
    plt.xlabel('X_1');
    plt.ylabel('X_2');
    plt.title('Overlayed Shadings According to Hyperplane');
    if show_c_1 == True:
        plt.show();
    plt.clf();

    classify_feature_space();
    plt.scatter(X_1, X_2, s = 75, color = color_labels);
    plt.title("Shading Highlighting Point Classification");
    plt.xlabel('X_1');
    plt.ylabel('X_2');
    plt.legend();
    if show_c_2 == True:
        plt.show();
    plt.clf();

    #Part d
    classify_feature_space(black_Z = False);

    x_in = (4*np.sqrt(5) + 4*np.sqrt(2) + 15)/(6 + 2*np.sqrt(5) + 2*np.sqrt(2));
    y_in = (4*np.sqrt(2) + np.sqrt(5) + 6)/(6 + 2*np.sqrt(5) + 2*np.sqrt(2));

    tri_linspace_1 = np.linspace(2, x_in, 100);
    tri_linspace_2 = np.linspace(x_in, 2.5, 100);

    beta_0_gr = 2 - (y_in - 2)/(x_in - 2)*2
    line_gr = ((y_in - 2)/(x_in - 2))*tri_linspace_1 + beta_0_gr;

    beta_0_gb = 0.5 - (y_in - 0.5)/(x_in - 2)*2;
    line_gb = ((y_in - 0.5)/(x_in - 2))*tri_linspace_1 + beta_0_gb;

    beta_0_rb = 1 - (y_in - 1)/(x_in - 2.5)*2.5;
    line_rb = ((y_in - 1)/(x_in - 2.5))*tri_linspace_2 + beta_0_rb;


    plt.fill_between(tri_linspace_1, line_gb, line_gr, color = 'green', linewidth = 3, alpha = 0.25);
    red_green_update_1 = 6 - 2*tri_linspace_1;
    plt.fill_between(tri_linspace_1, line_gr, red_green_update_1, color = 'red', linewidth = 3, alpha = 0.25);
    blue_green_update_1 = -1.5 + tri_linspace_1;
    plt.fill_between(tri_linspace_1, blue_green_update_1, line_gb, color = 'blue', linewidth = 3, alpha = 0.25);
    red_green_update_2 = 6 - 2*tri_linspace_2;
    plt.fill_between(tri_linspace_2, line_rb, red_green_update_2, color = 'red', linewidth = 3, alpha = 0.25);
    blue_green_udpate_2 = -1.5 + tri_linspace_2;
    plt.fill_between(tri_linspace_2, blue_green_udpate_2, line_rb, color = 'blue', linewidth = 3, alpha = 0.25);

    #part e
    X_1 = np.array([3,2,4,5,3,3,5,5,1,1,0, x_in]);
    X_2 = np.array([2,4,4,4,0,1,1,3,1,2,4, y_in]);
    color_labels = np.array(['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'green', 'green', 'green', 'black']);
    plt.scatter(X_1, X_2, color = color_labels);
    plt.title("Shading Highlighting Point Classification");
    plt.xlabel('X_1');
    plt.ylabel('X_2');
    if show_de == True:
        plt.show();
    plt.clf();




def Q4(show_a):
    #Putting Data in Matrix Form
    data = pd.read_csv("Netflix_Ratings.csv");
    movie_names = pd.read_csv("Netflix_Movies.csv", index_col = "Movie_ID");

    num_movies = np.unique(data['movie ID'].to_numpy())[-1];
    num_customers = np.unique(data['customer ID'].to_numpy())[-1];
    ratings = np.zeros(shape = (num_movies + 1, num_customers + 1));
    for i in np.unique(data["movie ID"].to_numpy()):
        for index, row in data.loc[data['movie ID'] == i].iterrows():
            ratings[row['movie ID']][row['customer ID']] = row['rating 1-5'];

    #part a
    if show_a == True:
        ratings[ratings == 0] = np.nan;

        #part a
        avg_rating = np.nanmean(ratings, axis = 1);
        worst_movie = (np.nanargmin(avg_rating), np.nanmin(avg_rating));
        worst_movie_name = movie_names['Name'][worst_movie[0]];
        best_movie = (np.nanargmax(avg_rating), np.nanmax(avg_rating));
        best_movie_name = movie_names['Name'][best_movie[0]];

        rated_or_not = ratings;
        rated_or_not[rated_or_not > 0] = 1;
        times_movie_rated = np.nansum(rated_or_not, axis = 1);
        least_rated = (np.argmin(times_movie_rated[1:]), np.min(times_movie_rated[1:]));
        least_rated_name = movie_names['Name'][least_rated[0] + 1];
        most_rated = (np.argmax(times_movie_rated), np.max(times_movie_rated));
        most_rated_name = movie_names['Name'][most_rated[0]];

        times_user_rated = np.nansum(rated_or_not, axis = 0);
        most_ratings = (np.argmax(times_user_rated), np.max(times_user_rated));
        least_ratings_indices = np.where(times_user_rated[1:] == np.min(times_user_rated[1:]))[0] + 1;
        least_ratings = np.min(times_user_rated[1:]);

        print('Name of Lowest Average Rated Movie: {}'.format(worst_movie_name));
        print('Rating of Lowest Average Rated Movie: {}'.format(worst_movie[1]));
        print();
        print('Name of Highest Average Rated Movie: {}'.format(best_movie_name));
        print('Rating of Highest Average Rated Movie: {}'.format(best_movie[1]));
        print();
        print('Name of Most Rated Movie: {}'.format(most_rated_name));
        print('Number of times this Movie was rated: {}'.format(int(most_rated[1])));
        print();
        print('Name of Least Rated Movie: {}'.format(least_rated_name));
        print('Number of times this Movie was rated: {}'.format(int(least_rated[1])));
        print();
        print('User ID that Rated the Most Movies: {}'.format(most_ratings[0]));
        print('Number of Movies this user Rated: {}'.format(int(most_ratings[1])));
        print();
        print('User ID(s) that Rated the Least Movies: {}'.format(least_ratings_indices));
        print('Number of Movies this user rated: {}'.format(int(least_ratings)));


    #part b
    ratings = ratings[1:, 1:];
    ratings = np.transpose(ratings);
    ratings_copy = np.copy(ratings);
    ratings_copy[ratings_copy == 0] = np.nan;
    x_ij = np.nonzero(ratings);
    x_zeros = np.where(ratings == 0);
    ratings_averages = np.nanmean(ratings_copy, axis = 0);

    ratings_df = pd.DataFrame(ratings_copy, index = [i for i in range(1,ratings_copy.shape[0] + 1)], columns = [j for j in range(1, ratings_copy.shape[1] + 1)]);
    for column in ratings_df:
        ratings_df[column].fillna((ratings_df[column].mean()), inplace = True);
    curr_iteration = 1;
    objective_values = [];
    while curr_iteration <= 500:
        pca = PCA(n_components = 10);
        pca.fit(ratings_df);
        A_hat = pca.transform(ratings_df);
        B_hat = pca.components_;
        replacement_vals = np.empty(shape = (len(x_zeros[0]),));
        for i in range(len(x_zeros[0])):
            replacement_vals[i] = np.dot(A_hat[x_zeros[0][i], :], B_hat[:, x_zeros[1][i]]) + ratings_averages[x_zeros[1][i]];
            if replacement_vals[i] > 5:
                replacement_vals[i] = 5;
            elif replacement_vals[i] < 1:
                replacement_vals[i] = 1;
            ratings_df[x_zeros[1][i] + 1][x_zeros[0][i] + 1] = replacement_vals[i];
        curr_objective_value = 0.0;
        for j in range(len(x_ij[0])):
            curr_objective_value = curr_objective_value + (ratings_df[x_ij[1][j] + 1][x_ij[0][j] + 1] - np.dot(A_hat[x_ij[0][j], :], B_hat[:, x_ij[1][j]]))**2
        print(curr_objective_value);
        #I beliueve this is Correct, but takes too long
        break;

#Calling Functions
Q3(show_a = True, show_b = True, show_c_1 = True, show_c_2 = True, show_de = True);
Q4(show_a = True);
