#모듈 임포트

import os
import sys
WORKING_DIR_AND_PYTHON_PATHS = os.path.join('/', *os.getcwd().split("/"))
# print(f'before {sys.path}')
sys.path.append(WORKING_DIR_AND_PYTHON_PATHS)
# print(f'after {sys.path}')


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#한글폰트
import platform
from matplotlib import font_manager, rc

import pickle as pickle
from opt import *
import time


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as lm
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split

# if platform.system() =='Darwin':
#     font_path = "/Library/Fonts/applegothic.ttf"
# elif platform.system() == 'Windows':
#     font_path = 'C₩'
# elif platform.system() == 'Linux':
# 	font_path = '/usr/share/fonts/open-sans/OpenSans-Regular.ttf'
#
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
#
#
# plt.style.use('seaborn')
# sns.set(font=font,
#         rc={"axes.unicode_minus":False},
#         style='darkgrid')


def main():
	print(f"{'='*10} start gird search {'='*10}")
	start_time = time.time()
	opt = parse_opts()
	print(f'- use model list {opt.models} -')

	# with open(os.path.join(opt.data_path,opt.file), "rb") as fh:
	# 	data= pickle.load(fh)

	try:
		dataset = pd.read_csv(os.path.join(opt.data_path, opt.file)).set_index('Index_ts')
		print(dataset.shape)
	except:
		print(f'<PathErr> check file path :{os.path.join(opt.data_path,opt.file)}')
		dataset = None

	X_feature = []
	y_feature = []
	X = dataset[X_feature]
	y = dataset[y_feature]
	X_train, y_train, X_val, y_val = train_test_split(X,y, test_size=0.1)

	if opt.modeltype =='ensemble':

		mapped_model = {'xgb':('xgboost', xgb.XGBRegressor()),
		'lr':('lr', lm.LinearRegression(n_jobs=-1)),
		'sgdr':('SGDRegressor', lm.SGDRegressor()),
		'ada': ('AdaBoostRegressor', AdaBoostRegressor()),
		'ridfe': ('ridge', lm.Ridge()),
		'lasso':('lasso', lm.Lasso()),
		 }
		n = 3

		params = {
		    'elastic': {
		        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], #default = 1.0
				'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], #default = 0.5
				'fit_intercept': [True, False], #default = True
				'normalize': [True, False], #default = False
				'selection': ['cyclic', 'random']  # default = 'cyclic'
				# 'precompute': [True, False, 'auto'],  # default = False
				# 'copy_X': [True, False], #default = True
				# 'warm_start': [True, False], #default = False
				# 'positive': [True, False], #default = False
				# 'random_state': [None, 0, 42], #default = None
				# 'max_iter': [100, 500, 1000, 5000, 10000],  # default = 1000
				# 'tol': [0.0001, 0.001, 0.01, 0.1, 1],  # default = 0.0001
		    },
		    'LassoLars': {
		        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], #default = 1.0
				'fit_intercept': [True, False], #default = True
		        'normalize': [True, False], #default = True
				# 'verbose': [True, False],  # default = False
				# 'precompute': [True, False, 'auto'],  # default = 'auto'
				# 'eps': [2.220446049250313e-16,], #default = np.finfo(float).eps
				# 'copy_X': [True, False], #default = True
				# 'fit_path': [True, False], #default = True
				# 'positive': [True, False], #default = False
				# 'jitter': [None], #default = None
				# 'random_state': [None, 0, 42] #default = None
				# 'max_iter': [100, 500, 1000, 5000, 10000],  # default = 500
		    },
		    'LogisticRegression': {
		        'penalty': ['l1', 'l2', 'elasticnet', 'none'], #default = 'l2'
		        'C': [0.01, 0.1, 1.0, 10, 100], #default = 1.0
		        'fit_intercept': [True, False], #default = True
				'class_weight': ['balanced', None], #default = None
				'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default = 'lbfgs'
				'max_iter': [10, 50, 100, 500, 1000, 5000], #default = 100
				# 'random_state': [None, 0, 42], #default = None
				# 'multi_class': ['auto', 'ovr','multinomial'], #default = 'auto'
				# 'verbose': [0, 1, 2, 3, 4 ,5, 6, 7, 8, 9, 10], #default = 0
				# 'warm_start': [True, False], #default = False
				# 'n_jobs': [None, -1, 1], #default = None
				# 'l1_ratio': [None, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #default = None
				# 'dual': [True, False], #default = False
				# 'tol': [0.0001, 0.001, 0.01, 0.1,1], #default = 0.0001
				# 'intercept_scaling': [0.1, 0.5, 1], #default = 1
		    },
		    'xgboost': {
				'eta': [0.05, 0.1, 0.15, 0.2, 0.3], #default = 0.3, learning_rate, Typical values 0.01~0.2
				"n_estimators": randint(100, 150),  # default = 100
				'max_depth': [3, 6, 9],  # default = 6, Typical values 3~10
				'min_child_weight': range(1, 6, 2),  # default = 1
				'gamma': uniform(0, 0.5).rvs(n), #default = 0
				'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], #default = 1, Typical values 0.5~1
				'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], #default = 1, Typical values 0.5~1
				'scale_pos_weight': 1, #default = 1
				'objective': 'binary:logistic',
				'booster': ['gbdt', 'dart'],
				'seed': 2021 #default = 0,
				# 'nthread': -1,
				# 'max_delta_step': [0], #default = 0, this parameter is generally not used
				# 'sampling_method': ['uniform', 'gradient_based'], #default = uniform
				# 'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1], #default = 1, not used often
				# 'lambda': [1,2,3], #default = 1, to reduce overfitting, not used often
				# 'alpha': [0], #default = 0
		    },
			'LGBM': {
				'learning_rate': [0.005, 0.01, 0.1], #default = 0.1
				'max_depth': range(2, 7),  # default = -1
				'num_leaves': [10, 20, 30, 40, 50 ,60], #2^(max_depth)보다 작은게 좋음 #default = 31
				'min_data_in_leaf':[20, 100, 500, 1000], #default = 100
				'boosting_type': ['gbdt', 'dart'], #default = 'gbdt'
				'n_estimators': randint(100, 150),  # default = 100
				'objective': 'regression', #default = 'regression'
				# 'early_stopping_round': [50],  # default = 0
				# 'lambda_l1': #default = 0
				# 'lambda_l2': #default = 0
				# 'min_gain_to_split' #default = 0
				# 'num_iterations': [100], #default = 100
				# 'device': 'cpu', #default = 'cpu'인데 gpu이용 시 gpu로 지정
				# 'lambda'
				# 'feature_fraction'
				# 'bagging_fraction'
			}
		}
		models = [mapped_model[model] for model in opt.models]
		print(models)
		
		best_model, best_mae = None, float('inf')
		for model_name, model in models:
		    param_grid = params[model_name]
		    grid = GridSearchCV(model, cv=5, n_jobs=-1, param_grid=param_grid)
		    grid = grid.fit(X_train, y_train)

		    model = grid.best_estimator_
		    predictions = model.predict(X_val)
		    mae = mean_absolute_error(y_val, predictions)

		    print(model_name, mae)
		    print(f'{model_name} MAE: {mae} \n best params {model}')

		    if mae < best_mae:
		        best_model = model
		


	elif opt.modeltype =='timeseries':
		import torch.nn as nn
		import torch.optim as optim

		mapped_model = {'lstm':('LSTM', nn.LSTM())}
		print(opt.modeltype)



	end_time = time.time()
	print(f'take {end_time-start_time:0.3f} s ')
	print(f"{'='*10} end gird search {'='*10}")
	return 

	

if __name__ =='__main__':
	main()



