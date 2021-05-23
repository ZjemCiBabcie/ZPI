import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def excel_to_python(file):
    dataset = pd.read_csv("{}".format(file))
    data = dataset.iloc[:, 1:]
    return data


def breaking_classification(dataframe):
    X = []
    y = []
    mean_X = statistics.mean(dataframe.iloc[:, 0].values)
    mean_Y = statistics.mean(dataframe.iloc[:, 1].values)
    lower_bound = mean_Y - 0.3
    upper_bound = mean_Y + 0.3

    aggro_breakpoint = 1.8

    i = 0
    temp = []
    if_current_lower_than = False
    for i in range(len(dataframe.index) - 1):
        current = None
        if len(temp) == 3:
            X.append(temp)
            difference = temp[2] - temp[0]
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


def training():
    X, y = breaking_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    # classifier_KNN.fit(X_train, y_train)
    #
    # classifier_SVM = SVC(kernel='rbf')
    # classifier_SVM.fit(X_train, y_train)

    classifier_decision_tree = DecisionTreeClassifier(criterion='entropy')
    classifier_decision_tree.fit(X_train, y_train)

    y_pred = classifier_decision_tree.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))


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
    df = excel_to_python("siedzenie_przodem.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    breaking_classification(df)
    training()

    # plot(df, data_normalised, 'NORMALISED')
    # plot(df, data_standarised, 'STANDARISED')
