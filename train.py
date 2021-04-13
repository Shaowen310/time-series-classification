# %%
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

# %%[markdown]
# ##Load Data
# %%
npy_files_train = []
files = glob.glob("./data/train/*.npy")
for i in tqdm(range(len(files))):
    npy_files_train.append(np.load("./data/train/" + str(i) + ".npy").T)

npy_files_test = []
files = glob.glob("./data/test/*.npy")
for i in tqdm(range(len(files))):
    npy_files_test.append(np.load("./data/test/" + str(i) + ".npy").T)

# %%[markdown]
# ## Read labels
# %%
label = np.genfromtxt('train_kaggle.csv', delimiter=',').T[1][1:]
n_samples_true = len(label[label == 1.0])
n_samples_false = len(label[label == 0.0])


# %%[markdown]
# ## Preprocess
# %%[markdown]
# ### Normalize
# %%
def normalize(npy_list, norm='max', mode='local'):
    npy_list_normalized = []
    for i in tqdm(range(len(npy_list))):
        data = np.copy(npy_list[i])
        rows = ~np.all(np.isnan(npy_list[i]), axis=1)
        if norm == 'max':
            if mode == 'local':
                non_zero_max_rows = np.abs(np.nanmax(npy_list[i][rows], axis=1)) >= 1e-5
                rows = np.nonzero(rows)[0][non_zero_max_rows]
                data[rows] = npy_list[i][rows] / np.expand_dims(
                    np.nanmax(npy_list[i][rows], axis=1), 1)
        npy_list_normalized.append(data)
    return npy_list_normalized


# %%
npy_files_train_normalized = normalize(npy_files_train)
npy_files_test_normalized = normalize(npy_files_test)


# %%[markdown]
# ## Fill missing
# %%
def fill_missing_by_neighbors_(npy_list):
    for i in tqdm(range(len(npy_list))):
        for j in range(40):
            df = pd.DataFrame(npy_list[i][j])
            df_f = df.fillna(method='ffill')
            df_b = df.fillna(method='bfill')
            df = (df_f + df_b) / 2
            npy_list[i][j] = df.to_numpy().T


def fill_missing_by_mean_(npy_list):
    for i in tqdm(range(len(npy_list))):
        for j in range(40):
            df = pd.DataFrame(npy_list[i][j])
            mean = df.mean()
            df = df.fillna(mean)
            npy_list[i][j] = df.to_numpy().T


# %%[markdown]
# ## Analysis
# %%
print(npy_files_train[0][0])

# %%
plt.subplot(111)
plt.imshow(npy_files_train[0], interpolation="None")
plt.colorbar()
plt.show()

# %%
plt.subplot(111)
plt.imshow(npy_files_train[8], interpolation="None")
plt.colorbar()
plt.show()

# %%
plt.subplot(111)
plt.imshow(npy_files_train[27], interpolation="None")
plt.colorbar()
plt.show()

# %%
i = 8
j = 29
np.polyfit(np.arange(len(npy_files_train[i][j])),
           np.nan_to_num(npy_files_train[i][j], np.nanmean(npy_files_train[8][15])), 1)


# %%[markdown]
# ## By variance and mean and max
# %%
def get_features(npy_files):
    feature_rows = np.arange(len(npy_files[0]))
    # ignore ID col
    feature_rows = np.delete(feature_rows, [2])
    n_base_features = len(feature_rows)
    n_features = 8 * n_base_features + 1
    df = np.empty((len(npy_files), n_features))
    for i in tqdm(range(len(npy_files))):
        data = npy_files[i][feature_rows]
        not_all_nan_rows = ~np.all(np.isnan(data), axis=1)
        var_features = np.zeros(n_base_features)
        var_features[not_all_nan_rows] = np.nanvar(data[not_all_nan_rows], axis=1)
        mean_features = np.zeros(n_base_features)
        mean_features[not_all_nan_rows] = np.nanmean(data[not_all_nan_rows], axis=1)
        max_features = np.zeros(n_base_features)
        max_features[not_all_nan_rows] = np.nanmax(data[not_all_nan_rows], axis=1)
        min_features = np.zeros(n_base_features)
        min_features[not_all_nan_rows] = np.nanmin(data[not_all_nan_rows], axis=1)
        polyfit_features = np.zeros((n_base_features, 2))

        data_nan_to_num = np.nan_to_num(data[not_all_nan_rows])
        polyfit_features[not_all_nan_rows] = np.polyfit(np.arange(data.shape[1]), data_nan_to_num.T,
                                                        1).T
        polyfit_features = polyfit_features.flatten('F')
        range_features = max_features - min_features
        peak_features = np.zeros(n_base_features)

        for j in range(n_base_features):
            peak_features[j] = len(find_peaks(data[j], height=0)) / data.shape[1]

        np_item = np.concatenate((var_features, mean_features, max_features, min_features,
                                  polyfit_features, range_features, peak_features))
        np_item = np.concatenate((np_item, [len(npy_files[i])]))
        df[i] = np_item
    return df


# %%
train_var = get_features(npy_files_train)
test_var = get_features(npy_files_test)

# %%
X_train, X_test, y_train, y_test = train_test_split(train_var,
                                                    label,
                                                    test_size=0.09,
                                                    random_state=42)

# %%[markdown]
# ## XGBoost
# %%
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'max_depth': 2, 'eta': 0.4, 'objective': 'binary:logistic'}
param['nthread'] = 8
param['eval_metric'] = 'auc'

num_round = 500
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)


# In[ ]:
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


dtest_real = xgb.DMatrix(test_var)
res = sigmoid(bst.predict(dtest_real, bst.best_ntree_limit))
np.arange(len(res))

col_names = ['Id', 'Predicted']
df = pd.DataFrame(columns=col_names)
df['Id'] = np.arange(len(res))
df['Predicted'] = res
df.to_csv('res.csv', index=False)

# %%
xgb.plot_importance(bst)

# %%[markdown]
# ## LightBGM
# %%
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

param = {'num_leaves': 48, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 340

bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
# %%
test_pred = bst.predict(test_var, num_iteration=bst.best_iteration)

col_names = ['Id', 'Predicted']
df = pd.DataFrame(columns=col_names)
df['Id'] = np.arange(len(test_pred))
df['Predicted'] = test_pred
df.to_csv('res.csv', index=False)
