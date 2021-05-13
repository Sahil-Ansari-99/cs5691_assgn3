import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


train_data = pd.read_csv('data/Dataset 1A/train.csv')
train_data = train_data.to_numpy()

dev_data = pd.read_csv('data/Dataset 1A/dev.csv')
dev_data = dev_data.to_numpy()


def get_class_wise_data(data):
    obj = {}
    d = len(data[0]) - 1
    for point in data:
        curr_class = point[d]
        if obj.get(curr_class) is None:
            obj[curr_class] = [point]
        else:
            obj.get(curr_class).append(point)
    for class_ in obj:
        obj[class_] = np.array(obj.get(class_))
    return obj


def get_pairwise_class_data(data, first_class, second_class):
    first_class_data = data.get(first_class)
    second_class_data = data.get(second_class)
    concat_data = np.concatenate((first_class_data, second_class_data))
    np.random.shuffle(concat_data)
    _, d = concat_data.shape
    y_ = concat_data[:, d-1]
    x_ = concat_data[:, :d-1]
    return x_, y_


def perceptron_model(x_, y_, alpha_=0.0001):
    model_ = Perceptron(tol=1e-3, random_state=0, alpha=alpha_)
    model_.fit(x_, y_)
    return model_


def linear_svm_model(x_, y_, c=1.0):
    model_ = SVC(C=c)
    model_.fit(x_, y_)
    return model_


def mlp(x_, y_, hidden_layers=5):
    model_ = MLPClassifier(hidden_layer_sizes=hidden_layers)
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


class_wise_data = get_class_wise_data(train_data)
val_class_wise_data = get_class_wise_data(dev_data)
x_data, y_data = get_pairwise_class_data(class_wise_data, 0, 2)
val_x_data, val_y_data = get_pairwise_class_data(val_class_wise_data, 0, 2)
# y_data = train_data[:, 2]
# x_data = train_data[:, :2]
# y_val = dev_data[:, 2]
# x_val = dev_data[:, :2]
#
# hidden_layer_sizes = [10, 15, 20]
# max_acc = 0
# best_size = 10
# for hidden_layer_size in hidden_layer_sizes:
#     model = mlp(x_data, y_data, hidden_layers=hidden_layer_size)
#     train_acc = model.score(x_data, y_data)
#     val_acc = model.score(x_val, y_val)
#     print(hidden_layer_size, 'Train Acc:', train_acc, 'Val Acc:', val_acc)
#     if val_acc > max_acc:
#         max_acc = val_acc
#         best_size = hidden_layer_size
#
# print('Best hidden layers:', best_size)
# model = mlp(x_data, y_data, hidden_layers=best_size)
# train_pred = predict(model, x_data)
# val_pred = predict(model, x_val)
# print('Train Confusion Matrix:')
# print(confusion_matrix(y_data, train_pred))
# print('Val Confusion Matrix:')
# print(confusion_matrix(y_val, val_pred))

cs = [0.1, 0.5, 1.0]
best_acc = 0
best_alpha = 0.01
for alpha in cs:
    model = linear_svm_model(x_data, y_data, c=alpha)
    train_acc = model.score(x_data, y_data)
    val_acc = model.score(val_x_data, val_y_data)
    print('C:', alpha, 'Train acc:', train_acc, 'Val acc:', val_acc)
    if val_acc >= best_acc:
        best_acc = val_acc
        best_alpha = alpha

# model = mlp(x_data, y_data, hidden_layers=best_size)
# support_vectors = model.support_vectors_
print('Best C:', best_alpha)
model = linear_svm_model(x_data, y_data, c=best_alpha)
train_pred = predict(model, x_data)
val_pred = predict(model, val_x_data)
print('Train Confusion Matrix:')
print(confusion_matrix(y_data, train_pred))
print('Val Confusion Matrix:')
print(confusion_matrix(val_y_data, val_pred))

# fig, ax = plt.subplots()
# colors = ['r', 'g', 'b', 'y']
# legend_arr = []
# i = 0
# min_x = 999
# max_x = -999
# min_y = 999
# max_y = -999
# for point in x_data:
#     if point[0] < min_x:
#         min_x = point[0]
#     if point[0] > max_x:
#         max_x = point[0]
#     if point[1] < min_y:
#         min_y = point[1]
#     if point[1] > max_y:
#         max_y = point[1]
# # ax.scatter(x, y, color=colors[i])
#
# x_list = np.linspace(min_x, max_x, 100)
# y_list = np.linspace(min_y, max_y, 100)
#
# z = np.zeros((len(x_list), len(x_list)))
# X = []
# Y = []
# for i in range(len(x_list)):
#     temp_x = []
#     temp_y = []
#     for j in range(len(y_list)):
#         temp_x.append(x_list[i])
#         temp_y.append(y_list[j])
#         point = np.array([x_list[i], y_list[j]])
#         z[i][j] = predict(model, point)
#     X.append(temp_x)
#     Y.append(temp_y)
#
# ax.contourf(X, Y, z)
#
# x = []
# y = []
# for point in x_data:
#     x.append(point[0])
#     y.append(point[1])
#     ax.scatter(x, y, color='k')
#     i += 1

# ax.scatter(support_vectors[:, 0], support_vectors[:, 1], color='r')

plt.show()
