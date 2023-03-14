import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = None
pd.options.display.max_columns = None

data = pd.read_csv(filepath_or_buffer='data/data_TPT.csv', encoding='UTF-8', low_memory=False)

pitchers = list(data['Pitcher'].unique())

pitchers_dict = {key: {'PitcherThrows': None, 'Pitches': None} for key in pitchers}

for pitcher in pitchers_dict.keys():
    pitchers_dict[pitcher]['PitcherThrows'] = list(data[data['Pitcher'] == pitcher]['PitcherThrows'].unique())
    pitchers_dict[pitcher]['Pitches'] = list(data[data['Pitcher'] == pitcher]['TaggedPitchType'].unique())

target_column = ['TaggedPitchType']
char_columns = ['Pitcher', 'PitcherThrows', 'PitcherTeam', 'Batter',
                'BatterSide', 'BatterTeam', 'PitcherSet', 'Top/Bottom', 'PitchCall', 'KorBB', 'HitType', 'PlayResult',
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

# Change to char columns
data[data.columns[data.columns.isin(char_columns)]] = data[data.columns[data.columns.isin(char_columns)]].astype('category')

# Change to int columns
data[data.columns[data.columns.isin(int_columns)]] = data[data.columns[data.columns.isin(int_columns)]].astype('Int64')

# Change to float columns
data[data.columns[data.columns.isin(num_columns)]] = data[data.columns[data.columns.isin(num_columns)]].astype(float)

id_feature = ['Pitcher']

num_features = ['RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'RelHeight', 'RelSide', 'Extension', 'PlateLocHeight', 'PlateLocSide',
            'ZoneSpeed', 'VertApprAngle', 'HorzApprAngle', 'yt_ZoneTime', 'yt_ReleaseDistance']

categorical_features = ['PitcherThrows', 'yt_ReleaseAccuracy', 'yt_ZoneAccuracy', 'Tilt']

# Select only features necessary for the model
data = data[target_column + id_feature + num_features + categorical_features]

# Remove NAN
data = data.dropna()

# Turn Cateogrical to dummies
data = pd.get_dummies(data, columns=categorical_features)
data['TaggedPitchType'] = data['TaggedPitchType'].replace(['Knuckleball', 'Splitter'], 'KN_or_SPLTR')

mask = data['PitcherThrows_Left'] == 1

data_left = data[mask]
data_right = data[~mask]

# Create X
X = pd.concat([data[id_feature + num_features], data[data.columns[data.columns.str.contains('PitcherThrows|Tilt|yt_ReleaseAccuracy|yt_ZoneAccuracy')]]], axis=1)
X = X.values

# The mask has to be on 13.

X_left = pd.concat([data_left[id_feature + num_features], data_left[data_left.columns[data_left.columns.str.contains('PitcherThrows|Tilt|yt_ReleaseAccuracy|yt_ZoneAccuracy')]]], axis=1)
X_left = X_left.values

X_right = pd.concat([data_right[id_feature + num_features], data_right[data_right.columns[data_right.columns.str.contains('PitcherThrows|Tilt|yt_ReleaseAccuracy|yt_ZoneAccuracy')]]], axis=1)
X_right = X_right.values

# Target Column
# y = data['TaggedPitchType']
y_left = data_left['TaggedPitchType']
y_right = data_right['TaggedPitchType']

# Convert categorical target variables to integers
# le = LabelEncoder()
# y = le.fit_transform(y)
le_l = LabelEncoder()
y_left = le_l.fit_transform(y_left)
le_r = LabelEncoder()
y_right = le_r.fit_transform(y_right)

Pitch_mask = {'Fastball': 0,
                    'Curveball': 1,
                    'Changeup': 2,
                    'Cutter': 3,
                    'Splitter': 4,
                    'Slider': 5,
                    'Sinker': 6,
                    'Knuckleball': 7}

Preds_translate = {0:'Fastball',
              1:'Curveball',
              2:'Changeup',
              3:'Cutter',
              4:'Splitter',
              5:'Slider',
              6:'Sinker',
              7:'Knuckleball'}

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_left, y_left, stratify=y_left, test_size=0.3, random_state=23)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_right, y_right, stratify=y_right, test_size=0.3, random_state=23)

# Define the pipeline steps
steps = [('scaler', StandardScaler()),
         ('classifier', OneVsRestClassifier(LogisticRegression(max_iter=1000)))]

# Create the pipeline
# pipeline = Pipeline(steps)
pipeline_l = Pipeline(steps)
pipeline_r = Pipeline(steps)

# Define the hyperparameter grid for GridSearchCV
param_grid = {'classifier__estimator__C': [0.1, 1, 10, 100]}

# Create a GridSearchCV object
# grid = GridSearchCV(pipeline, param_grid, cv=5)
grid_l = GridSearchCV(pipeline_l, param_grid, cv=5)
grid_r = GridSearchCV(pipeline_r, param_grid, cv=5)


# Fit the GridSearchCV object to the training data
# grid.fit(X_train, y_train)
grid_l.fit(X_train_l[:,1:], y_train_l)
grid_r.fit(X_train_r[:,1:], y_train_r)

# Get the best hyperparameters from the GridSearchCV
# best_params = grid.best_params_
best_params_l = grid_l.best_params_
best_params_r = grid_r.best_params_

# Define the function to predict only specified classes
def predict_classes(model, row, pitchers_dict):
    if row[0] in pitchers_dict.keys():
        possible_pitches = list(pitchers_dict[row[0]]['Pitches'])
        possible_pitches = [x for x in possible_pitches if str(x) != 'nan']
        X = row[1:].reshape(1, -1)
        probs = model.predict_proba(X)
        pred = model.classes_[np.argmax(probs, axis=1)]
        pred_2 = model.classes_[np.argsort(probs, axis=1)]
        pred_2 = [p[1] for p in pred_2]
        y_pred = np.vectorize(Preds_translate.get)(pred)
        if y_pred not in possible_pitches:
            # y_pred = np.vectorize(Preds_translate.get)(pred_2)
            y_pred = pred_2
        else:
            # y_pred = [p for p in pred if p in possible_pitches]
            y_pred = pred
    else:
        X = row[1:].reshape(1, -1)
        probs = model.predict_proba(X)
        y_pred = model.classes_[np.argmax(probs, axis=1)]
        # y_pred = np.vectorize(Preds_translate.get)(y_pred)
    return probs, y_pred

predictions = list([predict_classes(grid_l, row, pitchers_dict) for row in X_test_l])
probs_l = [prediction[0][0] for prediction in predictions]
preds_l = [prediction[1][0] for prediction in predictions]

predictions = list([predict_classes(grid_r, row, pitchers_dict) for row in X_test_r])
probs_r = [prediction[0] for prediction in predictions]
preds_r = [prediction[1] for prediction in predictions]
print('Preds', np.array(preds_l))
print('Y test', y_test_l)
# Predict the class probabilities on the test set
# probs = grid.predict_proba(X_test)
#probs_l_test = grid_l.predict_proba(X_test_l[:,1:])
# probs_r = grid_r.predict_proba(X_test_r[:,1:])

# Get the class predictions on the test set
# preds = grid.classes_[np.argmax(probs, axis=1)]
#preds_l_test = grid_l.classes_[np.argmax(probs_l, axis=1)]
# preds_r = grid_r.classes_[np.argmax(probs_r, axis=1)]
#print(preds_l_test)

# print(preds)
# print(np.vectorize(Preds_translate.get)(preds))
# print(preds_l)
# print(np.vectorize(Preds_translate.get)(preds_l))
# print(preds_r)
# print(np.vectorize(Preds_translate.get)(preds_r))

# Evalution of the model
from sklearn.metrics import accuracy_score
# accuracy_n = accuracy_score(y_test, preds)
# print('Accuracy :', accuracy_n)
accuracy_l = accuracy_score(list(y_test_l), preds_l)
print('Accuracy left model:', accuracy_l)
accuracy_r = accuracy_score(list(y_test_r), preds_r)
print('Accuracy right model:', accuracy_r)
quit()
from sklearn.metrics import confusion_matrix
# confusion_matrix_n = confusion_matrix(y_test, preds)
# print('Confusion matrix :', confusion_matrix_n)
confusion_matrix_l = confusion_matrix(y_test_l, preds_l)
print('Confusion matrix left model:', confusion_matrix_l)
confusion_matrix_r = confusion_matrix(y_test_r, preds_r)
print('Confusion matrix right model:', confusion_matrix_r)

from sklearn.metrics import classification_report
# classification_report_n = classification_report(y_test, preds, target_names=data['TaggedPitchType'].unique())
# print('Classification Report Normal:', classification_report_n)
classification_report_l = classification_report(y_test_l, preds_l, target_names=data_left['TaggedPitchType'].unique())
print('Classification Report Left:', classification_report_l)
classification_report_r = classification_report(y_test_r, preds_r, target_names=data_right['TaggedPitchType'].unique())
print('Classification Report Right:', classification_report_r)

from sklearn.metrics import f1_score
# f1_score_n = f1_score(y_test, preds, average='weighted')
# print('F1_Score Normal:', f1_score_n)
f1_score_l = f1_score(y_test_l, preds_l, average='weighted')
print('F1_Score Left:', f1_score_l)
f1_score_r = f1_score(y_test_r, preds_r, average='weighted')
print('F1_Score Right:', f1_score_r)

from sklearn.metrics import roc_auc_score
# auc_n = roc_auc_score(y_test, probs, multi_class="ovr")
# print('AUC Normal:', auc_n)
auc_l = roc_auc_score(y_test_l, probs_l, multi_class="ovr")
print('AUC Left:', auc_l)
auc_r = roc_auc_score(y_test_r, probs_r, multi_class="ovr")
print('AUC Right:', auc_r)




