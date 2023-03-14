# importing the required modules
import glob
import pandas as pd
import sweetviz as sv
import numpy

pd.options.display.max_rows = None
pd.options.display.max_columns = 150

data_frame = pd.DataFrame()

# specifying the path to csv files
path = "data/2021"

# csv files in the path
files = glob.glob(path + "/*.csv")

# defining an empty list to store
# content
content_2021 = []

# checking all the csv files in the
# specified path
for filename in files:

    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, index_col=None, sep=";")
    content_2021.append(df)

# specifying the path to csv files
path = "data/2022"

# csv files in the path
files = glob.glob(path + "/*.csv")

# defining an empty list to store
# content
content_2022 = []

# checking all the csv files in the
# specified path
for filename in files:

    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, index_col=None)
    content_2022.append(df)

# converting content to data frame
data_frame = pd.concat([pd.concat(content_2021), pd.concat(content_2022)])

# # Sweetviz
# my_report = sv.analyze(data_frame)
# my_report.show_html('Analyzing data.html')

# Columns to remove because of missing values.
remove_columns = ['PitcherId', 'BatterId', 'AutoPitchType',
                  'Notes', 'LastTrackedDistance', 'yt_SessionName',
                  'Note', 'CatcherId']

data_frame = data_frame.drop(columns=remove_columns)

data_frame.to_csv(path_or_buf='data/complete_data.csv', sep=',', index=False)
