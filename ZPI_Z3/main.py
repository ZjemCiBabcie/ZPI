from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter


def excel_to_python(file, sheet):
    dataset = pd.read_excel("{}".format(file), sheet_name=sheet)
    new_dataset = pd.DataFrame(columns=list('XYZ'))
    for i in range(dataset.iloc[:10000, 0].size):
        dataset.iloc[i, 0] = dataset.iloc[i, 0].replace(microsecond=0)
    k = 0
    while k <= dataset.iloc[:10000, 0].size:
        j = 0
        X = 0
        Y = 0
        Z = 0
        f = False
        while dataset.iloc[k, 0] == dataset.iloc[k + 1, 0]:
            f = True
            X += dataset.iloc[k, 1]
            Y += dataset.iloc[k, 2]
            Z += dataset.iloc[k, 3]
            j += 1
            k += 1
        if f is False:
            k += 1
        else:
            X += dataset.iloc[k, 1]
            Y += dataset.iloc[k, 2]
            Z += dataset.iloc[k, 3]
            j += 1
            new_dataset = new_dataset.append({'X': X/j, 'Y': Y/j, 'Z': Z/j}, ignore_index=True)
    return new_dataset


def smoothing(X):
    smoothed = X.apply(lambda x: savgol_filter(x, window_length=51, polyorder=3))
    return smoothed


def standardization(X):
    standardScaler = StandardScaler()
    X = standardScaler.fit_transform(X)
    return X


def normalisation(X):
    minmaxScaler = MinMaxScaler()
    X = minmaxScaler.fit_transform(X)
    return X


def feature_scaling(smoothed_data, action):
    if action == 'N':
        return normalisation(smoothed_data)
    elif action == 'S':
        return standardization(smoothed_data)


def plot(dataframe, scaled, title):
    fig, axs = plt.subplots(2)
    fig.suptitle("[1] RAW DATA, [2] {}".format(title))
    axs[0].plot(dataframe)
    axs[1].plot(scaled)
    fig.text(0.5, 0.04, 'Sequence number', ha='center')
    fig.text(0.08, 0.55, 'Value', va='center', rotation='vertical')
    plt.legend(['X', 'Y', 'Z'], loc='upper right', bbox_to_anchor=(1.07, 2))
    plt.savefig('{}.png'.format(title))
    plt.show()


if __name__ == "__main__":
    df = excel_to_python("opolska-brzezina.xlsx", "Acceleration")

    print(df)

    data_smoothed = smoothing(df)
    data_normalised = feature_scaling(data_smoothed, 'N')
    data_standarised = feature_scaling(data_smoothed, 'S')

    plot(df, data_normalised, 'NORMALISED')
    plot(df, data_standarised, 'STANDARISED')
