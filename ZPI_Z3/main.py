import pandas as pd
from sklearn.preprocessing import StandardScaler
import statistics
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matlab.engine


def run_matlab_in_python():
    eng = matlab.engine.start_matlab()
    eng.writing(nargout=0)
    eng.quit()


def excel_to_python(columns):
    training_set = pd.read_csv('training-set.csv')
    test_set = pd.read_csv('test-set.csv')
    training_data = training_set.iloc[:, columns:]
    test_data = test_set.iloc[:, columns:]
    return training_data, test_data


def breaking_feature_extraction(dataframe):
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


def acceleration_feature_extraction(dataframe):
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


def swerving_classification(dataframe):
    X = []
    y = []

    lower_bound = 10
    upper_bound = 30

    aggro_breakpoint = 2

    temp = []
    for i in range(len(dataframe.index) - 2):
        current = None
        if len(temp) == 5:
            X.append(temp)
            if max(temp) > aggro_breakpoint:
                y.append(1)
            else:
                y.append(0)
            temp = []
        if lower_bound < abs(dataframe.iloc[i + 1, 2] - dataframe.iloc[i, 2]) < upper_bound:
            temp.append(dataframe.iloc[i, 1])

    return np.array(X), np.array(y)


def training(training_set, test_set, maneuver, method):
    X_train, y_train, X_test, y_test = None, None, None, None

    if maneuver == 'BREAKING':
        X_train, y_train = breaking_feature_extraction(training_set)
        X_test, y_test = breaking_feature_extraction(test_set)
    elif maneuver == 'ACCELERATING':
        X_train, y_train = acceleration_feature_extraction(training_set)
        X_test, y_test = acceleration_feature_extraction(test_set)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    classifier = None

    if method == 'SVM':
        classifier = SVC(kernel='linear')
    elif method == 'TREE':
        classifier = DecisionTreeClassifier()
    elif method == 'NB':
        classifier = GaussianNB()
    else:
        print('Wrong name')

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return cm, accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    run_matlab_in_python()
    training_data, test_data = excel_to_python(columns=1)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(training_data)
    #
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(test_data)

    methods = ['SVM', 'TREE', 'NB']
    full_names = ['Support Vector Machine', 'Decision Tree', 'Naive Bayes']

    for i in range(len(methods)):
        cm, accuracy = training(training_data, test_data, 'BREAKING', method=methods[i])
        print("Confusion matrix for {} classifier is\n"
              " {}".format(full_names[i], cm))
        print("And accuracy score is {} ".format(accuracy))
        print(40*'=')
