import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm


train_data = pd.read_csv('data/Dataset 1B/train.csv')
train_data = train_data.to_numpy()

dev_data = pd.read_csv('data/Dataset 1B/dev.csv')
dev_data = dev_data.to_numpy()


def mlp(x_, y_, hidden_layers=(5, 5)):
    model_ = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=0, max_iter=250)
    model_.fit(x_, y_)
    return model_


def svm_model(x_, y_, kernel='rbf', degree=2):
    if kernel == 'polynomial':
        model_ = svm.SVC(kernel=kernel, degree=degree, random_state=0)
    else:
        model_ = svm.SVC(kernel=kernel, random_state=0)
    model_.fit(x_, y_)
    return model_


def predict(model_, x_):
    d = 0
    if len(x_.shape) == 1:
        d = len(x_)
    else:
        _, d = x_.shape
    x_ = x_.reshape(-1, d)
    y_ = model_.predict(x_)
    return y_


_, d_ = train_data.shape
y_data = train_data[:, d_-1]
x_data = train_data[:, :d_-1]
y_val = dev_data[:, d_-1]
x_val = dev_data[:, :d_-1]

# hidden_layer_sizes = [(20, 20), (25, 25), (30, 30), (40, 40)]
# max_acc = 0
# best_size = (20, 20)
# for hidden_layer_size in hidden_layer_sizes:
#     model = mlp(x_data, y_data, hidden_layers=hidden_layer_size)
#     acc = model.score(x_val, y_val)
#     print(hidden_layer_size, 'Acc:', acc)
#     if acc > max_acc:
#         max_acc = acc
#         best_size = hidden_layer_size

# model = mlp(x_data, y_data, hidden_layers=best_size)

model = svm_model(x_data, y_data, kernel='poly', degree=3)
support_vectors = model.support_vectors_
print(model.score(x_val, y_val))

fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'y']
legend_arr = []
i = 0
min_x = 999
max_x = -999
min_y = 999
max_y = -999
for point in x_data:
    if point[0] < min_x:
        min_x = point[0]
    if point[0] > max_x:
        max_x = point[0]
    if point[1] < min_y:
        min_y = point[1]
    if point[1] > max_y:
        max_y = point[1]
# ax.scatter(x, y, color=colors[i])

x_list = np.linspace(min_x, max_x, 100)
y_list = np.linspace(min_y, max_y, 100)

z = np.zeros((len(x_list), len(x_list)))
X = []
Y = []
for i in range(len(x_list)):
    temp_x = []
    temp_y = []
    for j in range(len(y_list)):
        temp_x.append(x_list[i])
        temp_y.append(y_list[j])
        point = np.array([x_list[i], y_list[j]])
        z[i][j] = predict(model, point)
    X.append(temp_x)
    Y.append(temp_y)

ax.contourf(X, Y, z)

x = []
y = []
for point in x_data:
    x.append(point[0])
    y.append(point[1])
    ax.scatter(x, y, color='k')
    i += 1

ax.scatter(support_vectors[:, 0], support_vectors[:, 1], color='r')

plt.show()
