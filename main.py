from PredictPositions.config import point_guards, training_players, testing_players, OPTIMAL_K
from PredictPositions.menu_functions import *
from PredictPositions.prediction_helper_functions import get_player_stats, get_career_avgs, get_player_id, get_player_position

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pickle


def create_training_data():
    '''() -> Numpy Array, list of strings
    This function creates the X and Y data to be used in training the KNN model,
    those being the per game career averages of each relevant stat for each player
    and the position that the player primarily played throughout their career.
    '''
    training_career_avgs, training_positions, testing_career_avgs, testing_positions = ([] for i in range(4))

    # for each player in the training and test sets, calculate and store the per game averages
    # for their career and their position
    for player in training_players:
        player_id = get_player_id(player)
        stats = get_player_stats(player_id)
        position = get_player_position(player_id)
        career_avgs = get_career_avgs(stats)
        training_positions.append(position)
        training_career_avgs.append(career_avgs)
    training_career_avgs = np.array(training_career_avgs)

    for player in testing_players:
        player_id = get_player_id(player)
        stats = get_player_stats(player_id)
        position = get_player_position(player_id)
        career_avgs = get_career_avgs(stats)
        testing_positions.append(position)
        testing_career_avgs.append(career_avgs)
    testing_career_avgs = np.array(testing_career_avgs)
    return (training_career_avgs, training_positions, testing_career_avgs, testing_positions)


if __name__ == '__main__':
    # either create and save the training data or load the training data if already created 
    try:
        with open("data.pickle", "rb") as f:
            [training_career_avgs, training_positions, testing_career_avgs, testing_positions] = pickle.load(f)
    except:
        training_career_avgs, training_positions, testing_career_avgs, testing_positions = create_training_data()
        with open("data.pickle", "wb") as f:
            pickle.dump([training_career_avgs, training_positions, testing_career_avgs, testing_positions], f)

    # encode the labels of the training data
    encoder = LabelEncoder()
    training_pos_encoded = np.array(encoder.fit_transform(training_positions))

    # create and fit the KNN model using the optimal value for k from the config.py file
    clf = KNeighborsClassifier(n_neighbors=OPTIMAL_K)
    clf.fit(training_career_avgs, training_pos_encoded)

    # continue to display the menu, prompt the user for a selection from the menu and continue with that selection until the user quits the program
    while True:
        inp = main_menu()
        if inp == '1':
            model_explanation(training_career_avgs, training_players, training_positions, len(point_guards))
        
        elif inp == '2':
            find_optimal_k(training_career_avgs, training_pos_encoded, testing_players, testing_career_avgs, testing_positions, encoder)

        elif inp == '3':
            _, _ = test_model(testing_players, testing_career_avgs, testing_positions, clf, encoder, show_results=True)

        elif inp == '4':
            predict_pos_user_name(training_career_avgs, training_players, training_positions, clf, encoder)
            pass

        elif inp == '5':
            predict_pos_user_stats(clf, encoder)

        elif inp == '0':
            break
