# NBA Position Prediction
This program uses NBA player statistics and K-Nearest Neighbours classification to predict the position of NBA players.

# Compatibility
Please note that this program is compatible with Python versions 3.7-3.9, as the scraper required installs Pandas 1.3.1 and Numpy 1.21.0. 

# Instructions
1. Download all files and ensure that the file structure is maintained.
2. Open a new terminal and change your current working directory to the NBAPositionPredicition folder downloaded in step 1.
3. It is recommended to create and activate a virtual environment before continuing.
3. Ensure that the below python libraries are installed by running the following:
   * pip install basketball-reference-scraper
   * pip install scikit-learn
   * pip install matplotlib
4. To start the program, run the following command:
   * python main.py

# Additional Note
You can alter the training algorithm by modifying the lists of players in config.py, then delete data.pickle and re-run the program to update the model.
