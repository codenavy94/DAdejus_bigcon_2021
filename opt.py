import argparse

def parse_opts():
	parser = argparse.ArgumentParser()

	parser.add_argument('--modeltype', type=str,
						default='ensemble', help='dataset tpye')

	parser.add_argument(
		'--root_path',
		default='/home/hail09/ML',
		type=str,
		help='Root dir path of code')

	parser.add_argument(
		'--data_path',
		default='./dataset',
		type=str,
		help='dir path of data')

	parser.add_argument(
		'--file',
		default='baseball_train_final4.csv',
		type=str,
		help='file name')

	parser.add_argument(
		'--windowsize',
		default=15,
		type=int,
		)

	parser.add_argument(
		'--num_workers',
		default=4,
		type=int,
		help='Number of threads for multi-thread loading')

	parser.add_argument(
		'--loss',
		default='mae',
		type=str,
		help='loss function')


	parser.add_argument(
		'-l',
		'--models',
		nargs='+',
		help=' add ensemble model  <ref> -l ada xgb lgbm cat ',
		# required=True
	)

	parser.add_argument(
		'--optim',
		default='adam',
		type=str,
		help='optim'
	)
	parser.add_argument(
		'--cri',
		default='',
	)

	parser.add_argument(
		'--y_feature',
		default='장타',
		type=str,
		help='y feature [장타 | 출루 | ops]'
	)

	parser.add_argument(
		'--use_cuda',
		action='store_true',
		help='If true, use GPU.')

	parser.add_argument(
		'--gpu',
		default=0,
		type=int)
	parser.add_argument(
		'--log_interval',
		default=10,
		type=int,
		help='Log interval for showing training loss')

	args = parser.parse_args()

	return args