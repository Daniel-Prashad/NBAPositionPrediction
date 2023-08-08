from PredictPositions.config import colours_five

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def get_coordinates(df):
    '''(Dataframe) -> Numpy Array
    This function takes a dataframe and returns a set of coordinates used to plot the sum of a player's scoring stats
    against the sum of their non-scoring stats
    '''
    # create and return the X and Y data points for each player, where each X is the sum of the given player's scoring
    # stats, while y is the sum of their non-scoring stats 
    scoring_stats = np.array(df[:,0]+df[:,4]+df[:,6])
    non_scoring_stats = np.array(df[:,1]+df[:,2]+df[:,3]+df[:,5])
    plot = np.stack((scoring_stats, non_scoring_stats), axis=1)
    return plot


def create_dataframe(plot, col_names, col_vals):
    '''(Numpy Array, List of Strings, List of Lists) -> Dataframe
    This function creates a dataframe which is then used to create a scatterplot.
    '''
    df = pd.DataFrame(plot, columns=['scoring_stats', 'non_scoring_stats'])
    for i, col_name in enumerate(col_names):
        df[col_name] = col_vals[i]
    return df


def get_legend_elements(colour_dict):
    '''(dict) -> List of Objects
    Given a dictionary, this function returns the legend elements with the appropriate labels and corresponding colours.
    '''
    legend_elements= []
    for pos in colour_dict:
        label_pos = pos
        legend_elements.append(Line2D([0], [0], marker='o', color='white', markerfacecolor=colour_dict[pos], label=label_pos))
    return legend_elements


def get_pos_avgs(training_career_avgs, size_per_pos):
    '''(Numpy Array, int) -> Numpy Array
    This function takes the career averages of each of the players in the training set, as well as the number of players
    in each position and returns the average of these career averages for each of the five positions.
    '''
    pos_avgs = []
    df = pd.DataFrame(training_career_avgs, columns =["AST", "STL", "BLK", "TRB", "FTA", "TOV", "3PA_to_2PA"])
    # for each of the five positions, calculate and store the average of each stat
    for i in range(0,5):
        rng = [i*size_per_pos, ((i+1)*size_per_pos)]
        curr_pos_avgs = np.array(round(df[rng[0]:rng[1]].mean(),1))
        pos_avgs.append(curr_pos_avgs)
    pos_avgs = np.array(pos_avgs)
    return pos_avgs


def visualize(training_career_avgs, training_players, training_positions, player_stats=None, player_name=""):
    '''(Numpy Array, List of Strings, List of Strings, Optional Numpy Array, Optional String) -> Nonetype
    This function visualizes the training data of the model using a scatterplot and the data point of a requested player.
    '''
    # get the coordinates for each data point
    plot = get_coordinates(training_career_avgs)

    # create a dataframe to store the data needed for the scatterplot
    df = create_dataframe(plot, ['player', 'position'], [training_players, training_positions])

    # define the scatterplot
    fig, ax = plt.subplots()
    
    # create the title and x- and y-labels for the graph
    fig.suptitle("NBA Positions Visualized") 
    ax.set(xlabel='Scoring Stats', ylabel='Non-Scoring Stats')

    # plot each point, using the appropriate colour for each point
    ax.scatter(df['scoring_stats'], df['non_scoring_stats'], c=df['position'].map(colours_five))

    # add data point for additional player if requested
    if player_stats is not None:
        plyr_scoring_stats = player_stats[0]+player_stats[4]+player_stats[6]
        plyr_non_scoring_stats = player_stats[1]+player_stats[2]+player_stats[3]+player_stats[5]
        ax.scatter(plyr_scoring_stats, plyr_non_scoring_stats, s=125, c="black", marker='s')
        ax.annotate(player_name, ((plyr_scoring_stats+0.20, plyr_non_scoring_stats+0.20)))

    # set the size of the plot
    plt.gcf().set_size_inches((14,10))

    # create the legend for the graph
    legend_elements_five = get_legend_elements(colours_five)
    ax.legend(handles=legend_elements_five, bbox_to_anchor=(1.0, 1.02), loc='upper left')

    # display the graphs
    plt.show()


def visualize_sample(training_career_avgs, training_players, training_positions, size_per_pos):
    '''(Numpy Array, List of Strings, List of Strings, int) -> Nonetype
    This function visualizes the average point of each of the five standard positions,
    as well as the points of five randomly selected players from each position.
    '''
    sample_indices = []
    n=5

    # store the indices of five randomly selected players for each position
    for i in range(0,5):
        sample_indices += random.sample(range(i*size_per_pos, ((i+1)*size_per_pos)-1), n)
    training_players_sample = [training_players[i] for i in sample_indices]
    training_positions_sample = [training_positions[i] for i in sample_indices]
    # get the coordinates for each data point of the randomly selected sample
    plot = get_coordinates(training_career_avgs[sample_indices])

    # create a dataframe to store the data needed for the scatterplot
    df = create_dataframe(plot, ['player', 'position'], [training_players_sample, training_positions_sample])

    # define the scatterplot
    fig, ax = plt.subplots()

    # for each of the five positions, calculate the average stats across all players of that position
    pos_avgs = get_pos_avgs(training_career_avgs, size_per_pos)

    # get the coordinates for each data point for the position averages
    pos_plot = get_coordinates(pos_avgs)

    # create a dataframe to store the data needed for the scatterplot
    pos_df = create_dataframe(pos_plot, ['player', 'position'], [['AVERAGE G', 'AVERAGE G/F', 'AVERAGE F', 'AVERAGE F/C', 'AVERAGE C'], ['GUARD', 'GUARD/FORWARD', 'FORWARD', 'FORWARD/CENTER', 'CENTER']])

    # plot each point, using the appropriate colour
    ax.scatter(pos_df['scoring_stats'], pos_df['non_scoring_stats'], s=1000, c=pos_df['position'].map(colours_five), marker='*', edgecolors='black')
    ax.scatter(df['scoring_stats'], df['non_scoring_stats'], c=df['position'].map(colours_five))

    # set the size of the plot
    plt.gcf().set_size_inches((10,10))

    # annotate both: each point representing the average of each position with the name of the corresponding position
    # and each of the points representing a player with the corresponding player's name
    for i in range(len(pos_df)):
        plt.annotate(pos_df['player'][i], ((pos_df['scoring_stats'][i]+0.05, pos_df['non_scoring_stats'][i]+0.05)))
    for i in range(len(df)):
        plt.annotate(df['player'][i], ((df['scoring_stats'][i]+0.05, df['non_scoring_stats'][i]+0.05)))

    # create the title and x- and y-labels for the graph
    plt.title("NBA Positions Visulaized")
    plt.xlabel("Scoring Stats")
    plt.ylabel("Non-Scoring Stats")
    
    # create the legend for the graph
    legend_elements = get_legend_elements(colours_five)      
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # display the graph
    plt.show()
    

def elbow_plot(exact_error_rates, adjusted_error_rates):
    '''(List of floats, List of floats) -> Nonetype
    This function outputs an elbow plot of the error rates from the evaluated models vs the corresponding k-values used.
    '''
    # plot the exact error rates as red and adjusted error rates as blue
    plt.figure()
    plt.plot(range(1,31), exact_error_rates, color='black', linestyle='dashed', marker='o', markerfacecolor='red', markersize=8)
    plt.plot(range(1,31), adjusted_error_rates, color='black', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=8)
    
    # create the title and x- and y-labels for the graph
    plt.title("Error Rate vs Number of Neighbours")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Error Rate")

    # create the legend for the graph
    legend_elements = [Line2D([0], [0], marker='o', color='white', markerfacecolor='red', label="Exact Error Rates"),
                       Line2D([0], [0], marker='o', color='white', markerfacecolor='blue', label="Adjusted Error Rates")]
    plt.legend(handles=legend_elements)

    # display the graph
    plt.show()
