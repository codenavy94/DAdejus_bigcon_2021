# loader.py




def makedataset(data):
	'''
	params: 
	'''
	feature_x = ['']
	feature_y = ['']
	X = data.loc[:,[]]

	return X, y

















class SeriesDataLoader(object):

    def __init__(self, rootpath='../dataset'):
        self.root_path = rootpath

    def load_data(self, file='ECG5000'):
        file_list = os.listdir(os.path.join(self.root_path, file))
        #re.compile('*.arff')

        raw_train = arff.loadarff('../dataset/ECG5000/ECG5000_TRAIN.arff')[0]
        test = pd.DataFrame(raw_train)

        raw_test = arff.loadarff('../dataset/ECG5000/ECG5000_TEST.arff')[0]
        train = pd.DataFrame(raw_test)


        # X = pd.concat([train, test], axis=0)
        # y = X.iloc[:, -1]
        df = pd.concat([train, test], axis=0)
        return df
        # return X.iloc[:, :-1], y
        # return train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1]


    def labeling(self,y):
        """

        :param y: pd.Series
        :return: list
        """

        label = {b'1': 'Normal', b'2': 'RonT', b'3': 'PVC', b'4': 'SP', b'5': 'UB'}
        return [label[_y] for _y in y]

    def train_test_split(self, X, y, test_size=0.15, shuffle=True, random_seed=103):
        np.random.seed = random_seed
        if X.index.array.shape[0] != y.index.array.shape[0]:
            raise ValueError(f'size of X {X.shape[0]}, y {y.shape[0]} is not same')

        if shuffle:
            _idx = X.index.array
            np.random.shuffle(_idx)
        else:
            _idx = X.index.array
        print()

        slice = int(len(_idx)*test_size)

        train_X = X.iloc[slice:, :]
        train_y = X.iloc[slice:, -1]

        val_X = X.iloc[:slice, :]
        val_y = X.iloc[:slice, -1]

        return train_X, train_y, val_X, val_y