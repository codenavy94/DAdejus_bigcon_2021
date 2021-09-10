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
		    'lr' : {
		        'fit_intercept': [True, False],
		        'normalize': [True, False],
		    },
		    'ridge': {
		        'alpha': [0.01, 0.1, 1.0, 10, 100],
		        'fit_intercept': [True, False],
		        'normalize': [True, False],
		    },
		    'lasso': {
		        'alpha': [0.1, 1.0, 10],
		        'fit_intercept': [True, False],
		        'normalize': [True, False],
		    },
		    'elastic': {
		        'alpha': [0.1, 1.0, 10],
		        'normalize': [True, False],
		        'fit_intercept': [True, False],
		    },
		    'LassoLars': {
		        'alpha': [0.1, 1.0, 10],
		        'normalize': [True, False],
		        'fit_intercept': [True, False],
		    },
		    'LogisticRegression': {
		        'penalty': ['l1', 'l2'],
		        'C': [0.001, 0.01, 0.1, 1.0, 10, 100],
		        'fit_intercept': [True, False],
		    },
		    'SGDRegressor': {
		        'penalty': ['l1', 'l2'],
		        'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100],
		        'fit_intercept': [True, False],
		    },
		    'Perceptron' :{
		        'penalty': ['None', 'l1', 'l2'],
		        'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100],
		        'fit_intercept': [True, False]
		    },
		    'xgboost': {
		        "gamma": uniform(0, 0.5).rvs(n),
		        "max_depth": range(2, 7), # default 3
		        "n_estimators": randint(100, 150).rvs(n), # default 100
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



