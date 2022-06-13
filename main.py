from PredictPositions.config import point_guards, shooting_guards, small_forwards, power_forwards, centers, training_players, testing_players
from PredictPositions.menu_functions import *
from PredictPositions.prediction_helper_functions import get_player_stats, get_career_avgs

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
    training_positions = []
    training_career_avgs = []
    # for each player in the training set, calculate and store the per game averages
    # for their career and their primary position
    for player in training_players:
        stats, primary_pos = get_player_stats(player)
        career_avgs = get_career_avgs(stats)
        training_positions.append(primary_pos)
        training_career_avgs.append(career_avgs)
    training_career_avgs = np.array(training_career_avgs)
    return training_career_avgs, training_positions


if __name__ == '__main__':
    # either create and save the training data or load the training data if already created 
    try:
        with open("data.pickle", "rb") as f:
            [training_career_avgs, training_positions] = pickle.load(f)
    except:
        training_career_avgs, training_positions = create_training_data()
        with open("data.pickle", "wb") as f:
            pickle.dump([training_career_avgs, training_positions], f)

    # encode the labels of the training data for the model with 5 positions, and create the labels of the training data for the moel with 3 positions
    encoder = LabelEncoder()
    training_pos_encoded = np.array(encoder.fit_transform(training_positions))
    y_train_3_pos = np.array([0]*(len(point_guards) + len(shooting_guards)) + [1]*(len(small_forwards) + len(power_forwards)) + [2]*(len(centers)))

    # create and fit both KNN models
    clf = KNeighborsClassifier()
    clf_3_pos = KNeighborsClassifier()
    clf.fit(training_career_avgs, training_pos_encoded)
    clf_3_pos.fit(training_career_avgs, y_train_3_pos)

    # continue to display the menu, prompt the user for a selection from the menu and continue with that selection until the user quits the program
    while True:
        inp = main_menu()
        if inp == '1':
            model_explanation(training_career_avgs, training_players, training_positions, y_train_3_pos, len(point_guards))
        
        elif inp == '2':
            test_model(testing_players, clf, clf_3_pos, encoder)

        elif inp == '3':
            predict_pos_user_name(training_career_avgs, training_players, training_positions, y_train_3_pos, clf, clf_3_pos, encoder)

        elif inp == '4':
            predict_pos_user_stats(clf, clf_3_pos, encoder)

        elif inp == '0':
            print("Goodbye\n")
            break
