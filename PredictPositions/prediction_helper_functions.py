from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
import pandas as pd
import numpy as np


def get_player_id(player_name):
    '''(str) -> str
    This function takes the name of an NBA player and returns their player_id, to be used to get their stats.
    '''
    player_id = players.find_players_by_full_name(player_name)[0]['id']
    return(player_id)


def get_player_stats(player_id):
    '''(str) -> dataframe
    This function takes the id of an NBA player and returns a dataframe of their average stats by year, used by the prediciton models.
    '''
    relevant_stats = ['AST', 'STL', 'BLK', 'REB', 'FTA', 'TOV', 'FGA', 'FG3A']
    player_stats = []
    # using the nba_api, pull the full record of stats for the given player
    career_stats = playercareerstats.PlayerCareerStats(player_id, per_mode36="PerGame").get_data_frames()[0]
    # drop the rows that contain a string in the 'G' (Games Played) column
    # if this column contains a string, it would be to describe the reason as to why the player did not play for that given year,
    # for example an injury, in this case, we can ignore this year for our calculations
    career_stats = career_stats[career_stats['GP'].apply(lambda x: str(x).replace('.','').isdigit())]
    # convert all relevant stats from strings to floats and calculate the ratio of 3-Pointers attempted to 2-Pointers attempted
    for stat in relevant_stats:
        career_stats[stat] = pd.to_numeric(career_stats[stat])
    career_stats['3PA_to_2PA'] = round(career_stats['FG3A'] / career_stats['FGA'], 2)
    # store and return the relevant stats of the given player
    player_stats = career_stats[["SEASON_ID", "AST", "STL", "BLK", "REB", "FTA", "TOV", "3PA_to_2PA"]]
    return(player_stats)


def get_player_position(player_id):
    '''(str) -> str
    This function takes the id of an NBA player and returns the position that they played throughout their career
    '''
    position = commonplayerinfo.CommonPlayerInfo(player_id).get_data_frames()[0]["POSITION"][0].upper()
    if position == "GUARD-FORWARD" or position == "FORWARD-GUARD":
        position = "GUARD/FORWARD"
    elif position == "FORWARD-CENTER" or position == "CENTER-FORWARD":
        position = "FORWARD/CENTER"
    return(position)


def get_career_avgs(stats):
    '''(Dataframe) -> Numpy Array
    This function takes a data frame of stats, where each row contains the per game average for each stat for a given year,
    and calculates and returns a numpy array of the per game averages of each stat throughout the player's career.
    '''
    # set the index of the data frame as the Season
    indexed = stats.set_index(['SEASON_ID'])
    # define a list of column values representing each of the relevant stats
    cols = [col for col in indexed.columns]
    # in the case that a player is traded during the season, the player will hae two entries (rows) with the same year
    # group the indices by the season so that the seasons in which a player was traded are combined as the mean of the rows
    season_avgs = round(indexed.groupby('SEASON_ID')[cols].mean(),1)
    # calculate and return the averages of each stat across each season in the dataframe
    career_avgs = np.array(round(season_avgs.mean(), 1))
    return career_avgs


def predict_position(player_name, clf, encoder):
    '''(str, KNN classifier, LabelEncoder) -> str, str, list of floats
    This function takes the name of an NBA player, predicts the position of that player
    and returns the prediction as well as the actual position that the player played throughout their career
    and a list of the player's career averages for relevant stats.
    '''
    # get the player's id, stats & primary position and calculate their per game averages throughout their career
    player_id = get_player_id(player_name)
    stats = get_player_stats(player_id)
    act_pos = get_player_position(player_id)
    career_avgs = get_career_avgs(stats).reshape(1,-1)
    # predict the position and return the predictions as well as the actual position played
    pred_pos_encoded = clf.predict(career_avgs)
    pred_pos = encoder.inverse_transform(pred_pos_encoded)[0]
    return pred_pos, act_pos, career_avgs[0]


def predict_position_from_stats(career_avgs, clf, encoder):
    '''(Numpy Array, KNN classifier, LabelEncoder) -> str
    Given a numpy array of career averages of each statistic, this function predicts and returns the position of this imaginary player.
    '''
    pred_pos_encoded = clf.predict(career_avgs)
    pred_pos = encoder.inverse_transform(pred_pos_encoded)[0]
    return pred_pos


def get_prediction_results(pred_pos, act_pos):
    '''(str, str) -> str
    This function takes the predicted and actual positions of a player and returns whether there is an exact, partial or no match.
    '''
    if pred_pos == act_pos:
        match_type = "Exact Match"
    elif pred_pos in act_pos or act_pos in pred_pos:
        match_type = "Partial Match"
    else:
        match_type = "No Match"
    return(match_type)
    