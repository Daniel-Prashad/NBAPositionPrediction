# NBA Position Prediction
This program uses NBA player statistics and K-Nearest Neighbours classification to predict the position of NBA players, using an elbow plot to find the opimal value for k.

# Compatibility
Please note that this program is compatible with Python versions => 3.7

# Instructions
1. Download all files and ensure that the file structure is maintained.
2. Open a new terminal and change your current working directory to the NBAPositionPredicition folder downloaded in step 1.
3. It is recommended to create and activate a virtual environment before continuing.
3. Ensure that the below python libraries are installed by running the following:
   * pip install nba_api
   * pip install pandas
   * pip install numpy
   * pip install requests
   * pip install scikit-learn
   * pip install matplotlib
4. To start the program, run the following command:
   * python main.py

# Additional Note
You can alter the training algorithm by modifying the lists of players or the OPTIMAL-K value in config.py, then delete data.pickle and re-run the program to update the model.