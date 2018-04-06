import random
from collections import defaultdict

import numpy as np

# Loading Datasets
print('Loading Datasets. \n')
X = np.load('data.npy')
class_index = np.load('class_a.npy').item()

train = []
test = []
for class_name, indexes in sorted(class_index.items(), key=lambda x: x[0]):
    random.shuffle(indexes)
    indice = len(indexes) // 10 * 9
    train_part = indexes[:indice]
    test_part = indexes[indice:]
    for x in train_part:
        train.append(x)
    for x in test_part:
        test.append(x)

X = np.array(X)
labels = np.array(labels)
width = X.shape[1]

X_train = X[train]
np.save('X_train', X_train)
labels_train = labels[train]
np.save('labels_train', labels_train)

X_test = X[test]
np.save('X_test', X_test)
labels_test = labels[test]
np.save('labels_test', labels_test)


for train_test in ['train', 'test']:
    # Loading Datasets
    print('Loading Datasets. \n')
    labels = np.load('labels_' + train_test + '.npy')

    class_index = defaultdict(list)
    for i, label in enumerate(labels):
        class_index[label].append(i)

    print('class_' + train_test + ' \n')
    np.save('class_' + train_test, class_index)
