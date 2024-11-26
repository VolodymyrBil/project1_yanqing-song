import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle


# parameters
n_splits = 5
n_estimators = 51
max_depth = 3
min_samples_leaf = 15

# data preparation
df = pd.read_csv(
    '/Users/yanqingsong/Documents/mypythonscript/practice scripts/ml zoomcamp/machine-learning-zoomcamp-homework/07-mid-project/heart_failure_clinical_records_dataset.csv'
)

df['age'] = df['age'].astype(int)
df.columns = df.columns.str.lower()

for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'death_event']:
    df[col] = df[col] == 1
df['sex'] = df['sex'].map({0: 'woman', 1: 'man'})

X_train_full, X_test, y_train_full, y_test = train_test_split(
    df.drop(columns='death_event'),
    (df['death_event']).astype(int),
    test_size=0.2,
    random_state=9,
)


# training
def train(X_train, y_train):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(X_train.to_dict(orient='records'))

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=7,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    return dv, rf_model


def predict(X, dv, model):
    X = dv.transform(X.to_dict(orient='records'))
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


kf = KFold(n_splits=n_splits, shuffle=True, random_state=7)
scores = []
fold = 0
for train_idx, val_idx in kf.split(X_train_full):
    X_train = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_train = y_train_full.iloc[train_idx]
    y_val = y_train_full.iloc[val_idx]

    dv, model = train(X_train, y_train)
    y_pred = predict(X_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'fold {fold} auc: {auc:.4f}')
    fold = fold + 1

print('\nvalidation auc:')
print(f'{np.mean(scores):.4f} ± {np.std(scores):.4f}')

print('\ntraining the final model')
dv, model = train(X_train_full, y_train_full)

y_pred = predict(X_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print(f'test auc = {auc:.4f}')

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print('\nmodel.bin saved.')
