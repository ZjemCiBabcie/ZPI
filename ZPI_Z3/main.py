import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.signal import savgol_filter


def excel_to_python(file, sheet):
    column_delete = "Timestamp"
    driverExcelFile = pd.ExcelFile("{}".format(file))
    df = pd.read_excel(driverExcelFile, "{}".format(sheet), usecols=lambda x: x not in column_delete)
    return df


def smoothing(dataframe):
    data_smoothed = dataframe.apply(lambda x: savgol_filter(x, window_length=51, polyorder=3))
    return data_smoothed


def standardization(dataframe):
    standscaler = StandardScaler()
    scaledscaler_df = standscaler.fit_transform(dataframe)
    standarized_df = pd.DataFrame(scaledscaler_df, columns=df.columns)
    return standarized_df


def normalization(dataframe):
    norm = Normalizer()
    norm_df = norm.fit_transform(dataframe)
    normalized_df = pd.DataFrame(norm_df, columns=df.columns)
    return normalized_df


def normalize_or_standarize(smoothed_data, action):
    if action == 'N':
        return normalization(smoothed_data)
    elif action == 'S':
        return standardization(smoothed_data)


def plot(dataframe, smoothed, normalized, title):
    fig, axs = plt.subplots(3)
    fig.suptitle("[1] DATA, [2] SMOOTHED, [3] {}".format(title))
    axs[0].plot(dataframe)
    axs[1].plot(smoothed)
    axs[2].plot(normalized)
    fig.text(0.5, 0.04, 'Sequence number', ha='center')
    fig.text(0.08, 0.55, 'Value', va='center', rotation='vertical')
    plt.legend(['X', 'Y', 'Z'], loc='upper right', bbox_to_anchor=(1.07, 3.45))
    plt.show()


if __name__ == "__main__":
    df = excel_to_python("kierowcaA.xlsx", "Acceleration")
    print(df)

    data_smoothed = smoothing(df)
    data_normalized = normalize_or_standarize(data_smoothed, 'N')
    data_standarized = normalize_or_standarize(data_smoothed, 'S')

    plot(df, data_smoothed, data_normalized, 'NORMALIZED')
    plot(df, data_smoothed, data_standarized, 'STANDARIZED')
