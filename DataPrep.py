import pandas as pd
import numpy as np
import datetime



pd.options.display.max_rows = None
pd.options.display.max_columns = None


data = pd.read_csv('data/complete_data.csv', encoding='UTF-8')
print(data.head())

date_columns = ['Date']
time_columns = ['Time']
char_columns = ['Pitcher', 'PitcherThrows', 'PitcherTeam', 'Batter',
                'BatterSide', 'BatterTeam', 'PitcherSet', 'Top/Bottom', 'TaggedPitchType', 'PitchCall', 'KorBB', 'HitType', 'PlayResult',
                'HomeTeam', 'AwayTeam', 'Stadium', 'Level', 'League', 'GameID', 'PitchUUID', 'yt_ZoneAccuracy', 'Catcher', 'CatcherTeam',
                'Tilt', 'yt_ReleaseAccuracy', 'yt_ZoneAccuracy']
int_columns = ['PitchNo', 'PAofInning', 'PitchofPA', 'Inning', 'Outs', 'Balls', 'Strikes', 'OutsOnPlay', 'RunsScored']
num_columns = ['RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis', 'RelHeight', 'RelSide', 'Extension', 'VertBreak',
               'InducedVertBreak', 'HorzApprAngle', 'ZoneTime', 'ExitSpeed', 'Angle', 'Direction', 'HitSpinRate', 'PositionAt110X',
               'PositionAt110Y', 'PositionAt110Z', 'Distance', 'Bearing', 'HangTime', 'pff', 'pfxz', 'x0', 'y0', 'z0', 'vx0',
               'vy0', 'vz0', 'ax0', 'ay0', 'az0', 'yt_RelSpeed', 'yt_RelHeight', 'yt_RelSide', 'yt_VertRelAngle', 'yt_HorzRelAngle',
               'yt_ZoneSpeed', 'yt_PlateLocHeight', 'yt_PlateLocSide', 'yt_VertApprAngle', 'yt_HorzApprAngle', 'yt_ZoneTime', 'yt_HorzBreak',
               'yt_InducedVertBreak', 'yt_OutOfPlane', 'yt_FSRI', 'yt_EffectiveSpin', 'yt_GyroSpin', 'yt_Efficiency', 'yt_SpinComponentX',
               'yt_SpinComponentY', 'yt_SpinComponentZ', 'yt_HitVelocityX', 'yt_HitVelocityY', 'yt_HitVelocityZ', 'yt_HitLocationX',
               'yt_HitLocationY', 'yt_HitLocationZ', 'yt_GroundLocationX', 'yt_GroundLocationY', 'yt_HitBreakX', 'yt_HitBreakY',
               'yt_HitBreakT', 'yt_HitSpinComponentX', 'yt_HitSpinComponentY', 'yt_HitSpinComponentZ', 'yt_PitchSpinConfidence',
               'yt_PitchReleaseConfidence', 'yt_HitSpinConfidence', 'yt_EffectiveBattingSpeed', 'yt_SeamLat',
               'yt_SeamLong', 'yt_ReleaseDistance']

# Change Date format
data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y").strftime("%Y-%m-%d"))

# Change Time format
data['Time'] = pd.to_datetime(data['Time']).dt.strftime('%H:%M:%S')

# Change to char columns
data[data.columns[data.columns.isin(char_columns)]] = data[data.columns[data.columns.isin(char_columns)]].astype('category')

# Change to int columns
data[data.columns[data.columns.isin(int_columns)]] = data[data.columns[data.columns.isin(int_columns)]].astype('Int64')

# Change to float columns
data[data.columns[data.columns.isin(num_columns)]] = data[data.columns[data.columns.isin(num_columns)]].astype(float)

# Select Specified columns for TaggedPitchType model (to replace AutoPitchType)
print(data.columns)
columns_to_drop = ['PitchNo', 'Date', 'Time', 'PAofInning', 'Inning', 'Top/Bottom', 'Outs', 'Balls', 'Strikes',
                   'OutsOnPlay', 'RunsScored', 'HomeTeam', 'AwayTeam', 'Stadium', 'Level', 'League', 'GameID',
                   'PitchUUID', 'Catcher', 'CatcherTeam']

data_TPT = data.drop(columns=columns_to_drop)

# Save to csv
data_TPT.to_csv(path_or_buf='data/data_TPT.csv', encoding='UTF-8', index=False)