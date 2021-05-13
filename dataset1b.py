import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('data/Dataset 1B/train.csv')
train_data = train_data.to_numpy()

dev_data = pd.read_csv('data/Dataset 1B/dev.csv')
dev_data = dev_data.to_numpy()


def mlp(x_, y_, hidden_layers=(5, 5), activation_='relu', max_iter=250):
    model_ = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=0, max_iter=max_iter, activation=activation_)
    model_.fit(x_, y_)
    return model_


def svm_model(x_, y_, kernel='rbf', degree_=2, c_=1.0):
    if kernel == 'poly':
        model_ = svm.SVC(kernel=kernel, degree=degree_, random_state=0)
    else:
        model_ = svm.SVC(kernel=kernel, C=c_, random_state=0)
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

# hidden_layer_sizes = [(20, 20), (30, 30), (40, 40), (70, 70)]
# activations = ['relu', 'logistic', 'tanh']
# max_acc = 0
# best_size = (20, 20)
# best_activation = 'relu'
# for hidden_layer_size in hidden_layer_sizes:
#     for activation in activations:
#         model = mlp(x_data, y_data, hidden_layers=hidden_layer_size, activation_=activation)
#         train_acc = model.score(x_data, y_data)
#         val_acc = model.score(x_val, y_val)
#         print(hidden_layer_size, activation, 'Train Acc:', train_acc, 'Val Acc:', val_acc)
#         if val_acc > max_acc:
#             max_acc = val_acc
#             best_size = hidden_layer_size
#             best_activation = activation
#
# print('Best size:', best_size, 'Best Activation:', best_activation)
# model = mlp(x_data, y_data, hidden_layers=best_size, activation_=best_activation)
# train_pred = predict(model, x_data)
# val_pred = predict(model, x_val)
# print('Train Confusion Matrix:')
# print(confusion_matrix(y_data, train_pred))
# print('Val Confusion Matrix:')
# print(confusion_matrix(y_val, val_pred))

model = mlp(x_data, y_data, hidden_layers=(40, 40), activation_='relu', max_iter=250)
weights = model.coefs_
intercepts = model.intercepts_
test_data = x_data[0]
node_num = 3
layer_num = 3
# for weight in weights[:2]:
#     test_data = np.matmul(weight.T, test_data)
#     print(weight.shape)

# degrees = [2, 3, 4, 5, 6]
# cs = [0.01, 0.1, 1.0, 10]
# best_degree = 3
# best_acc = 0
# for degree in cs:
#     model = svm_model(x_data, y_data, kernel='rbf', c_=degree)
#     train_acc = model.score(x_data, y_data)
#     val_acc = model.score(x_val, y_val)
#     print(degree, 'Train Acc:', train_acc, 'Val Acc:', val_acc)
#     if val_acc >= best_acc:
#         best_acc = val_acc
#         best_degree = degree
#
# print('Best C:', best_degree)
# model = svm_model(x_data, y_data, kernel='rbf', c_=best_degree)
# train_pred = predict(model, x_data)
# val_pred = predict(model, x_val)
# print('Train Confusion Matrix:')
# print(confusion_matrix(y_data, train_pred))
# print('Val Confusion Matrix:')
# print(confusion_matrix(y_val, val_pred))
# support_vectors = model.support_vectors_
# print(model.score(x_val, y_val))
#
# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
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
        b = 0
        for weight in weights[:layer_num]:
            point = np.matmul(weight.T, point) + intercepts[b]
            b += 1
            ACTIVATIONS['relu'](point)
        # z[i][j] = predict(model, point)
        z[i][j] = point[node_num - 1]
    X.append(temp_x)
    Y.append(temp_y)

ax.plot_surface(X, Y, z)

# x = []
# y = []
# for point in x_data:
#     x.append(point[0])
#     y.append(point[1])
#     ax.scatter(x, y, color='k')
#     i += 1

# ax.scatter(support_vectors[:, 0], support_vectors[:, 1], color='r')

plt.show()
