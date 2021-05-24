import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def excel_to_python(columns):
    training_set = pd.read_csv('training-set.csv')
    test_set = pd.read_csv('test-set.csv')
    training_data = training_set.iloc[:, columns:]
    test_data = test_set.iloc[:, columns:]
    print(training_data)
    return training_data, test_data


def breaking_classification(dataframe):
    X = []
    y = []
    mean_Y = statistics.mean(dataframe.iloc[:, 1].values)
    lower_bound = mean_Y - 0.3
    upper_bound = mean_Y + 0.3

    aggro_breakpoint = 1.8

    temp = []
    if_current_lower_than = False
    for i in range(len(dataframe.index) - 1):
        current = None
        if len(temp) == 3:
            X.append(temp)
            difference = temp[-1] - temp[0]
            if abs(difference) > aggro_breakpoint:
                y.append(1)
            else:
                y.append(0)
            temp = []
            if_current_lower_than = False
        if dataframe.iloc[i, 1] < lower_bound or dataframe.iloc[i, 1] > upper_bound:
            current = dataframe.iloc[i, 1]
            if len(temp) == 1:
                if temp[0] > current:
                    if_current_lower_than = True
                    temp.append(dataframe.iloc[i, 1])
                else:
                    temp = [dataframe.iloc[i, 1]]
                    continue
            elif len(temp) > 1:
                if temp[-1] > current and if_current_lower_than is True:
                    temp.append(dataframe.iloc[i, 1])
                else:
                    temp = []
                    if_current_lower_than = False
                    temp.append(current)
            elif len(temp) == 0:
                temp.append(dataframe.iloc[i, 1])

    return np.array(X), np.array(y)


def acceleration_classification(dataframe):
    X = []
    y = []
    mean_Y = statistics.mean(dataframe.iloc[:, 1].values)
    lower_bound = mean_Y - 0.3
    upper_bound = mean_Y + 0.3

    aggro_breakpoint = 1.8

    temp = []
    if_current_greater_than = False
    for i in range(len(dataframe.index) - 1):
        current = None
        if len(temp) == 3:
            X.append(temp)
            difference = temp[-1] - temp[0]
            if abs(difference) > aggro_breakpoint:
                y.append(1)
            else:
                y.append(0)
            temp = []
            if_current_greater_than = False
        if dataframe.iloc[i, 1] < lower_bound or dataframe.iloc[i, 1] > upper_bound:
            current = dataframe.iloc[i, 1]
            if len(temp) == 1:
                if temp[0] < current:
                    if_current_greater_than = True
                    temp.append(dataframe.iloc[i, 1])
                else:
                    temp = [dataframe.iloc[i, 1]]
                    continue
            elif len(temp) > 1:
                if temp[-1] < current and if_current_greater_than is True:
                    temp.append(dataframe.iloc[i, 1])
                else:
                    temp = []
                    if_current_greater_than = False
                    temp.append(current)
            elif len(temp) == 0:
                temp.append(dataframe.iloc[i, 1])

    return np.array(X), np.array(y)


def turning_classification(dataframe):
    X = []
    y = []
    mean_X = statistics.mean(dataframe.iloc[:, 0].values)
    mean_Y = statistics.mean(dataframe.iloc[:, 1].values)

    lower_bound_X = mean_X - 0.15
    upper_bound_X = mean_X + 0.15

    lower_bound_Y = mean_Y - 0.3
    upper_bound_Y = mean_Y + 0.3

    aggro_breakpoint = 1.8

    # temp = []
    # if_current_greater_than = False
    # for i in range(len(dataframe.index) - 1):
    #     current = None
    #     if len(temp) == 3:
    #         X.append(temp)
    #         difference = temp[-1] - temp[0]
    #         if abs(difference) > aggro_breakpoint:
    #             y.append(1)
    #         else:
    #             y.append(0)
    #         temp = []
    #         if_current_greater_than = False
    #     if dataframe.iloc[i, 1] < lower_bound or dataframe.iloc[i, 1] > upper_bound:
    #         current = dataframe.iloc[i, 1]
    #         if len(temp) == 1:
    #             if temp[0] < current:
    #                 if_current_greater_than = True
    #                 temp.append(dataframe.iloc[i, 1])
    #             else:
    #                 temp = [dataframe.iloc[i, 1]]
    #                 continue
    #         elif len(temp) > 1:
    #             if temp[-1] < current and if_current_greater_than is True:
    #                 temp.append(dataframe.iloc[i, 1])
    #             else:
    #                 temp = []
    #                 if_current_greater_than = False
    #                 temp.append(current)
    #         elif len(temp) == 0:
    #             temp.append(dataframe.iloc[i, 1])

    return np.array(X), np.array(y)


def training(training_set, test_set, maneuver, method):
    X_train, y_train, X_test, y_test = None, None, None, None

    if maneuver == 'B':
        X_train, y_train = breaking_classification(training_set)
        X_test, y_test = breaking_classification(test_set)
    elif maneuver == 'A':
        X_train, y_train = acceleration_classification(training_set)
        X_test, y_test = breaking_classification(test_set)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    classifier = None

    if method == 'KNN':
        classifier = KNeighborsClassifier()
    elif method == 'TREE':
        classifier = DecisionTreeClassifier(criterion='entropy')
    elif method == 'SVM':
        classifier = SVC(kernel='linear')
    elif method == 'FOREST':
        classifier = RandomForestClassifier(criterion='entropy')
    elif method == 'NB':
        classifier = GaussianNB()
    elif method == 'LR':
        classifier = LogisticRegression()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return cm, accuracy_score(y_test, y_pred)


def smoothing(data):
    smoothed = data.apply(lambda x: savgol_filter(x, window_length=51, polyorder=3))
    return smoothed


def plot(dataframe, scaled, title):
    fig, axs = plt.subplots(2)
    fig.suptitle("[1] RAW DATA, [2] {}".format(title))
    axs[0].plot(dataframe)
    axs[1].plot(scaled)
    fig.text(0.5, 0.04, 'Sequence number', ha='center')
    fig.text(0.08, 0.55, 'Value', va='center', rotation='vertical')
    plt.legend(['X', 'Y'], loc='upper right', bbox_to_anchor=(1.07, 2))
    plt.savefig('{}.png'.format(title))
    plt.show()


if __name__ == "__main__":
    training_data, test_data = excel_to_python(columns=1)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(training_data)
    #
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(test_data)

    turning_classification(training_data)

    # methods = ['KNN', 'TREE', 'SVM', 'FOREST', 'NB', 'LR']
    # full_names = ['K-nearest Neighbors', 'Decision Tree', 'Support Vector Machine', 'Random Forest', 'Naive Bayes', 'Linear Regression']
    #
    # for i in range(len(methods)):
    #     cm, accuracy = training(training_data, test_data, 'B', method=methods[i])
    #     print("Confusion matrix for {} classifier is\n"
    #           " {}".format(full_names[i], cm))
    #     print("And accuracy score is {} ".format(accuracy))
    #     print(40*'=')
