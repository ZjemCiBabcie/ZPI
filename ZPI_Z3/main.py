import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter


def excel_to_python(file, sheet):
    dataset = pd.read_excel("{}".format(file), sheet_name=sheet)
    X = dataset.iloc[:, 1:]
    return X


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
    df = excel_to_python("kierowcaA.xlsx", "Acceleration")
    print(df)

    data_smoothed = smoothing(df)
    data_normalised = feature_scaling(data_smoothed, 'N')
    data_standarised = feature_scaling(data_smoothed, 'S')

    plot(df, data_normalised, 'NORMALISED')
    plot(df, data_standarised, 'STANDARISED')
