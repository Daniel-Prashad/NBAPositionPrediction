from PredictPositions.plot_helper_functions import visualize, visualize_sample, elbow_plot
from PredictPositions.prediction_helper_functions import predict_position, predict_position_from_stats, get_prediction_results
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import os


def main_menu():
    '''() -> string
    This function displays the menu options to the user, prompts for and returns the user's selection of one of these options.
    '''
    # define the list of options
    options = ['1', '2', '3', '4', '5', '0']

    # display the menu
    os.system('cls')
    print("Welcome to Predicting NBA Positions!")
    print("[1] - Explanation of the model")
    print("[2] - View the elbow plot of Error Rates vs Number of Neighbours used to determine the optimal k-value")
    print("[3] - View the results of a preselected test set of 60 players")
    print("[4] - Provide a player's name to predict their position")
    print("[5] - Provide statistics to predict a player's position")
    print("[0] - Exit program")

    # prompt for, store and return the user's selection, ensuring that the input is valid
    run = True
    while run:
        inp = input("Please select from one of the above options ([1], [2], [3], [4] or [0]): ")
        if inp in options:
            run = False
    os.system('cls')
    return inp


def model_explanation(training_career_avgs, training_players, training_positions, size_per_pos):
    '''(Numpy Array, List of Strings, List of Strings, Numpy Array, int) -> Nonetype
    This function provides the user with an explanation of the model, visualizes the training data, provides inferences, etc.
    '''
    print("MODEL EXPLANATION")
    print("In order to predict a player's position, this model uses the K-Nearest Neighbours algorithm.")
    print("We start with a list of 200 NBA players, with a 70/30 split; that is 140 players in the training set and 60 players in the test set.")
    print("The career averages for the following stats are then calculated for each player:")
    print("     Categorized as 'SCORING STATS' are: Assists, Free Throw Attempts, Ratio of 3-Pointers Attempted to 2-Pointers Attempted")
    print("     Categorized as 'NON-SCORING STATS' are: Steals, Blocks, Total Rebounds and Turnovers")
    print("     (The players in the training set also have the position that they primarily played throughout their career stored for use in the next point)")
    print("For each player training set, the sum of their SCORING STATS is plotted against the sum of their NON-SCORING STATS and the point is labelled accordingly.")
    print("This completes training the model.\n")

    print("Now, in predicting the position of a player we calculate their career averages and plot the point as we did above.")
    print("Next, we look at the six closest points (nearest neighbours) to the one just plotted.")
    print("This point (and its corresponding player) is classified identically to the majority of those six nearest neighbours.")
    print("To see why we chose to look at the six nearest neighbours, select option [2] from the main menu. ")
  
    print("Please take a look at the generated graph.\n")
    print("Here we see that points labelled guard, forward or center generally reside within their own region.")
    print("The trend that we see is that the ratio of scoring stats to non-scoring stats is generally higher in guards, lower in centers and in-between for forwards.")
    print("As expected, players labelled as guard/forward generally fall in the region between guards and forwards, as does players labelled as forward/center fall between forwards and centers.")
    print("Taking this overlap into account, when validating the predicitons made by the model, we will consider an exact match to be worth 1.")
    print("While a partial match (where prediciton=guard for an actual guard or guard/forward, or the other way around) is worth 0.5.\n")
    print("Please close the graph to continue.\n")
    visualize(training_career_avgs, training_players, training_positions)

    print("The next graph plots the average point of each of the five positions, as well as the data points of five randomly selected players for each position from the training set.")
    print("This graph, as well as the previous one, shows that guards generally have more scoring stats than non-scoring stats, while centers tend to have the opposite.")
    print("The average of each position in between, those being the guard/forward, forward and forward/center, show a gradual transition from guard to center, respectively.\n")
    print("Please close the graph to return to the main menu.")
    visualize_sample(training_career_avgs, training_players, training_positions, size_per_pos)


def find_optimal_k(training_career_avgs, training_pos_encoded, testing_players, testing_career_avgs, testing_positions, encoder):
    '''(Numpy Array, Numpy Array, List of Strings, Numpy Array, Numpy Array, LabelEncoder) -> Nonetype
    This function evaluates our model for values of k between 1 and 30 and produces an elbow plot of the error rates vs k-value
    to determine the optimal value of k.
    '''
    # define the lists to store the error rates
    exact_error_rates  = []
    adjusted_error_rates = []
    
    # for k = 1, ... , 30 define, fit and evaluate the model, storing the error rates for each
    for i in range (1, 31):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(training_career_avgs, training_pos_encoded)
        exact_score, adjusted_score = test_model(testing_players, testing_career_avgs, testing_positions, clf, encoder)
        exact_error_rates.append(round(1-exact_score, 2))
        adjusted_error_rates.append(round(1-adjusted_score, 2))
        print(f"Number of Neighbours: {i} | Exact Score: {exact_score} | Adjusted Score: {adjusted_score}")
    
    # using the calculated error rates, output the elbow plot of error rates vs k-values
    input("\nPress [Enter] to view the elbow plot showing the Error Rates vs Number of Neighbours: ")
    print("\nFrom the elbow plot, we see that the optimal value for k is 6 because k=6 is at the \'elbow\' of the graph.")
    print("This means that k=6 is the minimal number of neighbours to consider in our classification that will maximize accuracy without the risk of over-fitting.")
    print("\nPlease close the graph to return to the main menu.")
    elbow_plot(exact_error_rates, adjusted_error_rates)


def test_model(testing_players, testing_career_avgs, testing_positions, clf, encoder, show_results = False):
    '''(List of Strings, Numpy Array, Numpy Array, KNN Classifier, LabelEncoder, Boolean) -> float, float
    This function is used to test the trained model, built from a preselected set of 60 players.
    It returns the exact and adjusted scores of the evaluated model.
    '''
    # create variable to store the number of correct predictions
    correct_preds = 0
    partial_preds = 0

    # predict the position for each of the 60 players in the test set
    # dislay the results and increment the number of each model if it correctly predicted the current player
    for i in range(len(testing_players)):
        career_avgs = testing_career_avgs[i].reshape(1,-1)
        pred_pos = predict_position_from_stats(career_avgs, clf, encoder)
        act_pos = testing_positions[i]
        match_type = get_prediction_results(pred_pos, act_pos)
        if match_type == "Exact Match":
            correct_preds += 1
        elif match_type == "Partial Match":
            partial_preds += 0.5
        if show_results:
            print(f"Name: {testing_players[i]} | Predicted Position: {pred_pos} | Actual Position: {act_pos} | Match Type: {match_type}\n")

    # display the number of correct predicitions and accuracy for exact and exact + partial matches
    exact_score = round(correct_preds/len(testing_players), 2)
    adjusted_score = round((correct_preds+partial_preds)/len(testing_players), 2)
    if show_results:
        print(f"Number of Exact Correct Predictions: {correct_preds}/{len(testing_players)} | Accuracy: {exact_score * 100}%")
        print(f"Number of Adjusted Correct Predictions: {correct_preds + partial_preds}/{len(testing_players)} | Accuracy: {adjusted_score * 100}%\n")
        input("Press [Enter] to return to the main menu: ")
    return(exact_score, adjusted_score)


def predict_pos_user_name(training_career_avgs, training_players, training_positions, clf, encoder):
    '''(Numpy Array, List of Strings, List of Strings, KNN classifier, LabelEncoder) -> Nonetype
    This function prompts the user for the name of an NBA player, predicts position of that player and displays the results to the user.
    '''
    run = True
    while run:
        # prompt and store the user for the name of a player
        inp_name = input("\nPlease enter the name of a NBA player (or 'exit' to return to the main menu): ")
        # return to the main menu if the user chooses
        if inp_name.lower() == 'exit':
            run = False
            break
        else:
            # otherwise, if the name is of a valid NBA player, predict the user's postion and display the results
            try:
                pred_pos, act_pos, career_avgs = predict_position(inp_name, clf, encoder)
                match_type = get_prediction_results(pred_pos, act_pos)
                print(f"Name: {inp_name} | Predicted Position: {pred_pos} | Actual Position: {act_pos} | Match Type: {match_type}")
                print(f"Please see the generated graph where {inp_name} is represented by the black sqaure.")
                print("Please close the graph to continue.")
                visualize(training_career_avgs, training_players, training_positions, career_avgs, inp_name)
            # if the name is invalid, prompt the user for another name
            except:
                print(f"I could not find {inp_name}'s stats. Please try a different player.")


def predict_pos_user_stats(clf, encoder):
    '''(KNN classifier, LabelEncoder) -> Nonetype
    This function prompts the user for some stats for an imginary NBA player, predicts position of that player and displays the predictions to the user.
    '''
    # define the stats that will be needed from the users and an empty list to store these stats
    stats = ["Assists", "Steals", "Blocks", "Total Rebounds", "Free Throw Attempts", "Turnovers", "2-Pointers Attempted", "3-Pointers Attempted"]
    career_avgs = []

    # for each of the defined stats, prompt the user for input, ensure that it is a valid float,
    # round it to two decimal places and append it to the list of stats
    print("For each of the following statistics, please enter this imaginary player's career average per game as a decimal number.")
    for stat in stats:
        while True:
            inp_stat = input(f"{stat}: ")
            try:
                inp_stat = round(float(inp_stat), 2)
                break
            except:
                print("Please enter a decimal number")
        career_avgs.append(inp_stat)

    # calculate the final statistic for each model, the ratio of 3-Pointers attempted to 2-Pointers attempted,
    # remove the stats used in this calculation from the list and append this ratio
    career_avgs[6] = 1 if career_avgs[6] == 0 else career_avgs[6]
    ratio_3P_to_2P = round((career_avgs[7]/career_avgs[6]), 2)
    career_avgs.pop()
    career_avgs.pop()
    career_avgs.append(ratio_3P_to_2P)

    # convert the list to a np array, predict the position and display the results to the user
    career_avgs = np.array(career_avgs).reshape(1,-1)
    pred_pos = predict_position_from_stats(career_avgs, clf, encoder)
    print(f"\nPredicted Prediction: {pred_pos}\n")
    input("Press [Enter] to return to the main menu: ")
