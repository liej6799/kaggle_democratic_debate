# variables
num_of_train_data = 2500

# entry point of the program
import pandas as pd

# DO NOT CHANGE THIS
pd.set_option('display.max_columns', None)
# DO NOT CHANGE THIS

# load the data
data = pd.read_csv('data/debate_transcripts.csv')

print(data['speaking_time_seconds'].isnull().any())

