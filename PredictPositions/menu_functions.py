from PredictPositions.config import three_pos_dict
from PredictPositions.plot_helper_functions import visualize, visualize_sample
from PredictPositions.prediction_helper_functions import predict_position, predict_position_from_stats

import numpy as np
import os


def main_menu():
    '''() -> string
    This function displays the menu options to the user, prompts for and returns the user's selection of one of these options.
    '''
    # define the list of options
    options = ['1', '2', '3', '4', '0']

    # display the menu
    os.system('cls')
    print("Welcome to Predicting NBA Positions!")
    print("[1] - Explanation of the model")
    print("[2] - View the results of a preselected test set of 60 players")
    print("[3] - Provide a player's name to predict their position")
    print("[4] - Provide statistics to predict a player's position")
    print("[0] - Exit program")

    # prompt for, store and return the user's selection, ensuring that the input is valid
    run = True
    while run:
        inp = input("Please select from one of the above options ([1], [2], [3], [4] or [0]): ")
        if inp in options:
            run = False
    print()
    return inp


def model_explanation(training_career_avgs, training_players, training_positions, y_train_3_pos, size_per_pos):
    '''(Numpy Array, List of Strings, List of Strings, Numpy Array, int) -> Nonetype
    This function provides the user with an explanation of the model, visualizes the training data, provides inferences, etc.
    '''
    #os.system('cls')
    print("MODEL EXPLANATION")
    print("In order to predict a player's position, this model uses the K-Nearest Neighbours algorithm.")
    print("We start with a list of 200 NBA players, with a 70/30 split; that is 140 players in the training set and 60 players in the test set.")
    print("The career averages for the following stats are then calculated for each player:")
    print("     Categorized as 'SCORING STATS' are: Assists, Free Throw Attempts, Ratio of 3-Pointers Attempted to 2-Pointers Attempted")
    print("     Categorized as 'NON-SCORING STATS' are: Steals, Blocks, Total Rebounds and Turnovers")
    print("     (The players in the training set also have the position that they primarily played throughout their career stored for use in the next point).")
    print("For each player training set, the sum of their SCORING STATS is plotted against the sum of their NON-SCORING STATS and the point is labelled accordingly.")
    print("This completes training the model.\n")

    print("Now, in predicting the position of a player we calculate the averages and plot the point as we did above.")
    print("Next, we look at the five closest points (nearest neighbours) to the one just plotted.")
    print("This point (and its corresponding player) is classified identically to the majority of those five nearest neighbours.")
 
    print("The problem with this model is that the players in the modern-age NBA often play more than one position, some even play all five!")
    print("In today's NBA, the guard positions have been combined, as have the forward positions.")
    print("Players are now usually classified as either a guard, forward or center.")
    print("Changing our model to reflect this will increase its accuracy in predicting a player's position.")
    print("Please take a look at the generated graphs.\n")
    print("In the top graph notice the signficant amount of overlap between classes.")
    print("This overlap and resulting grey area in the predictions of our model are primarily caused by two things:")
    print("     Firstly, although a player may be considered to be, and primarily play as, a certain position, often times they can play other positions as well.")
    print("     Secondly, the traditional roles and emphases of each position has changed over time.")
    
    print("Consequently, by referring to five nearest neighbours we can see that a lot of shooting guards can be mislabelled as either point guards or small forwards,\nand a lot of power forwards can be mislabelled as either small forwards or centers.")
    print("In the bottom graph, using the modern day three positions instead, we see that this overlap between classes is significantly reduced.")
    print("As a result, using these three classes instead of the traditional five positions, our model will be much less prone to misclassification.\n")
    print("Please close the graph to continue.\n")
    visualize(training_career_avgs, training_players, training_positions, y_train_3_pos,)

    print("The next graph plots the average point of each of the five positions, as well as the data points of five randomly selected players for each position from the training set")
    print("This graph, as well as the previous one, shows that point guards generally have more scoring stats than non-scoring stats,\nwhile centers tend to have the opposite.")
    print("The average of each position in between, those being the shooting guard, small forward and power forward, show a gradual transition from point guard to center, respectively")
    print("However, this graph better depicts the results of the aforementioned problem, as the randomly selected data points are not concentrated around their respective position average.\n")
    
    print("The predicitions made in this program will show the results of two models.")
    print("The first will classify a player as one of the five traditional positions: PG, SG, SF, PF or C")
    print("And the second will classify a player as one of the three modern day positions: GUARD, FORWARD or CENTER")
    print("In general, the latter model will be more accurate.\n")
    print("Please close the graph to return to the main menu.")
    visualize_sample(training_career_avgs, training_players, training_positions, size_per_pos)


def test_model(testing_players, clf, clf_3_pos, encoder):
    '''(List of Strings, KNN Classifier, KNN Classifier, LabelEncoder) -> Nonetype
    This function is used to test both trained models, built from a preselected set of 60 players.
    '''
    # create variables to store the number of correct predictions for both models
    correct_preds = 0
    correct_3_preds = 0

    # predict the position for each of the 60 players in the test set, using both models
    # dislay the results and increment the number of each model if it correctly predicted the current player
    for player in testing_players:
        pred_pos, pred_3_pos, act_pos, _ = predict_position(player, clf, clf_3_pos, encoder)
        print(f"Name: {player} | 5-Pos Prediction: {pred_pos} | 3-Pos Prediction: {three_pos_dict.get(pred_3_pos)[0]} | Actual Position: {act_pos} | 5-Pos Match: {pred_pos == act_pos} | 3-Pos Match: {act_pos in three_pos_dict.get(pred_3_pos)[1]}\n")
        if pred_pos == act_pos:
            correct_preds += 1
        if act_pos in three_pos_dict.get(pred_3_pos)[1]:
            correct_3_preds += 1

    # display the number of correct predicitions and accuracy for each model
    print(f"5-Pos -> Number of Correct Predictions: {correct_preds}/{len(testing_players)} | Accuracy: {round(correct_preds/len(testing_players), 2) * 100}%")
    print(f"3-Pos -> Number of Correct Predictions: {correct_3_preds}/{len(testing_players)} | Accuracy: {round(correct_3_preds/len(testing_players), 2) * 100}%")
    input("Press enter to return to the main menu")


def predict_pos_user_name(training_career_avgs, training_players, training_positions, y_train_3_pos, clf, clf_3_pos, encoder):
    '''(Numpy Array, List of Strings, List of Strings, Numpy Array, KNN classifier, KNN classifier, LabelEncoder) -> Nonetype
    This function prompts the user for the name of an NBA player, predicts position of that player
    (using both models) and displays the predictions to the user.
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
            # otherwise, if the name is of a valid NBA player, predict the user's postions using both models and display the predictions
            try:
                pred_pos, pred_3_pos, act_pos, career_avgs = predict_position(inp_name, clf, clf_3_pos, encoder)
                print(f"Name: {inp_name} | 5-Pos Prediction: {pred_pos} | 3-Pos Prediction: {three_pos_dict.get(pred_3_pos)[0]} | Actual Position: {act_pos} | 5-Pos Match: {pred_pos == act_pos} | 3-Pos Match: {act_pos in three_pos_dict.get(pred_3_pos)[1]}")
                print(f"\nPlease see the generated graph where {inp_name} is represented by the black sqaure.")
                print("Please close the graph to continue.")
                visualize(training_career_avgs, training_players, training_positions, y_train_3_pos, career_avgs, inp_name)
            # if the name is invalid, prompt the user for another name
            except:
                print(f"I could not find {inp_name}'s stats. Please try a different player.")


def predict_pos_user_stats(clf, clf_3_pos, encoder):
    '''(KNN classifier, KNN classifier, LabelEncoder) -> Nonetype
    This function prompts the user for some stats for an imginary NBA player, predicts position of that player
    (using both models) and displays the predictions to the user.
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
    pred_pos, pred_3_pos = predict_position_from_stats(career_avgs, clf, clf_3_pos, encoder)
    print(f"\n5-Position Prediction: {pred_pos}")
    print(f"3-Position Prediction: {three_pos_dict[pred_3_pos][0]}")
    input("Press enter to return to the main menu")
