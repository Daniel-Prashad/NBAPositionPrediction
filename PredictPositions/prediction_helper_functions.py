from basketball_reference_scraper.players import get_stats

import pandas as pd
import numpy as np


def get_player_stats(player):
    '''(str) -> dataframe, string
    This function takes the name of an NBA player and returns a dataframe of stats used by the prediciton models,
    as well as the position that the player primarily played throughout their career.
    '''
    stats = []
    # using the basketball reference scrapper, pull the full record of stats for the given player
    career_stats = get_stats(player)
    # drop the rows that contain a string in the 'G' (Games Played) column
    # if this column contains a string, it would be to describe the reason as to why the player did not play for that given year,
    # for example an injury, in this case, we can ignore this year for our calculations
    career_stats = career_stats[career_stats['G'].apply(lambda x: str(x).replace('.','').isdigit())]
    # convert all relevant stats from strings to floats and calculate the ratio of 3-Pointers attempted to 2-Pointers attempted
    career_stats['AST'] = pd.to_numeric(career_stats['AST'])
    career_stats['STL'] = pd.to_numeric(career_stats['STL'])
    career_stats['BLK'] = pd.to_numeric(career_stats['BLK'])
    career_stats['TRB'] = pd.to_numeric(career_stats['TRB'])
    career_stats['FTA'] = pd.to_numeric(career_stats['FTA'])
    career_stats['TOV'] = pd.to_numeric(career_stats['TOV'])
    career_stats['2PA'] = pd.to_numeric(career_stats['2PA'])
    career_stats['3PA'] = pd.to_numeric(career_stats['3PA'])
    career_stats['3PA_to_2PA'] = round(career_stats['3PA'] / career_stats['2PA'], 2)
    # store and return the relevant stats and primary position played by the given player
    stats = career_stats[["SEASON", "AST", "STL", "BLK", "TRB", "FTA", "TOV", "3PA_to_2PA"]]
    primary_pos = career_stats["POS"].value_counts().idxmax()
    return stats, primary_pos


def get_career_avgs(stats):
    '''(Dataframe) -> Numpy Array
    This function takes a data frame of stats, where each row contains the per game average for each stat for a given year,
    and calculates and returns a numpy array of the per game averages of each stat throughout the player's career.
    '''
    # set the index of the data frame as the Season
    indexed = stats.set_index(['SEASON'])
    # define a list of column values representing each of the relevant stats
    cols = [col for col in indexed.columns]
    # in the case that a player is traded during the season, the player will hae two entries (rows) with the same year
    # group the indices by the season so that the seasons in which a player was traded are combined as the mean of the rows
    season_avgs = round(indexed.groupby('SEASON')[cols].mean(),1)
    # calculate and return the averages of each stat across each season in the dataframe
    career_avgs = np.array(round(season_avgs.mean(), 1))
    return career_avgs


def predict_position(player, clf, clf_3_pos, encoder):
    '''(str, KNN classifier, KNN classifier, LabelEncoder) -> str, int, str, list of floats
    This function takes the name of an NBA player, predicts the position of that player using both models
    and returns both predictions as well as the actual primiary position that the player played throughout their career
    and a list of the player's career averages for relevant stats.
    '''
    # get the player's stats & primary position and calculate their per game averages throughout their career
    stats, act_pos = get_player_stats(player)
    career_avgs = get_career_avgs(stats).reshape(1,-1)
    # predict the position using both models and return the predictions as well as the actual position played
    pred_pos_encoded = clf.predict(career_avgs)
    pred_pos = encoder.inverse_transform(pred_pos_encoded)[0]
    pred_3_pos = clf_3_pos.predict(career_avgs)[0]
    return pred_pos, pred_3_pos, act_pos, career_avgs[0]


def predict_position_from_stats(career_avgs, clf, clf_3_pos, encoder):
    '''(Numpy Array, KNN classifier, KNN classifier, LabelEncoder) -> str
    Given a numpy array of career averages of each statistic, this function predicts and returns the position of this imaginary player.
    '''
    pred_pos_encoded = clf.predict(career_avgs)
    pred_pos = encoder.inverse_transform(pred_pos_encoded)[0]
    pred_3_pos = clf_3_pos.predict(career_avgs)[0]
    return pred_pos, pred_3_pos
    