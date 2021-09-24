import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statistics
import numpy as np
from scipy.signal import savgol_filter
import matlab.engine
import datetime


def run_matlab_in_python():
    eng = matlab.engine.start_matlab()
    eng.writing(nargout=0)
    eng.quit()


def excel_to_python():
    acceleration = pd.read_csv('acceleration.csv')
    speed = pd.read_csv('speed.csv')
    return acceleration, speed


def standarisation(dataframe):
    sc = StandardScaler()
    scaled_df = sc.fit_transform(dataframe)
    standarised_df = pd.DataFrame(scaled_df, columns=dataframe.columns)
    return standarised_df


def normalisation(dataframe):
    norm = MinMaxScaler()
    norm_df = norm.fit_transform(dataframe)
    normalised_df = pd.DataFrame(norm_df, columns=dataframe.columns)
    return normalised_df


def plot(dataframe, scaled, title):
    fig, axs = plt.subplots(2)
    fig.suptitle('[1] RAW DATA, [2] {}'.format(title))
    axs[0].plot(dataframe)
    axs[1].plot(scaled)
    fig.text(0.5, 0.04, 'Sequence number', ha='center')
    fig.text(0.08, 0.55, 'Value', va='center', rotation='vertical')
    plt.legend(['Y'], loc='upper right', bbox_to_anchor=(1.07, 3.45))
    plt.savefig('{}.jpg'.format(title))
    plt.show()


def breaking_feature_extraction(dataframe, I, P):
    X = []
    y = []
    mean_Y = statistics.mean(dataframe.iloc[:, 0].values)
    lower_bound = mean_Y - I
    upper_bound = mean_Y + I

    aggro_breakpoint = P

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
        if dataframe.iloc[i, 0] < lower_bound or dataframe.iloc[i, 0] > upper_bound:
            current = dataframe.iloc[i, 0]
            if len(temp) == 1:
                if temp[0] > current:
                    if_current_lower_than = True
                    temp.append(dataframe.iloc[i, 0])
                else:
                    temp = [dataframe.iloc[i, 0]]
                    continue
            elif len(temp) > 1:
                if temp[-1] > current and if_current_lower_than is True:
                    temp.append(dataframe.iloc[i, 0])
                else:
                    temp = []
                    if_current_lower_than = False
                    temp.append(current)
            elif len(temp) == 0:
                temp.append(dataframe.iloc[i, 0])

    return np.array(X), np.array(y)


def acceleration_feature_extraction(dataframe, I, P):
    X = []
    y = []
    mean_Y = statistics.mean(dataframe.iloc[:, 0].values)
    lower_bound = mean_Y - I
    upper_bound = mean_Y + I

    aggro_breakpoint = P

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
        if dataframe.iloc[i, 0] < lower_bound or dataframe.iloc[i, 0] > upper_bound:
            current = dataframe.iloc[i, 0]
            if len(temp) == 1:
                if temp[0] < current:
                    if_current_greater_than = True
                    temp.append(dataframe.iloc[i, 0])
                else:
                    temp = [dataframe.iloc[i, 0]]
                    continue
            elif len(temp) > 1:
                if temp[-1] < current and if_current_greater_than is True:
                    temp.append(dataframe.iloc[i, 0])
                else:
                    temp = []
                    if_current_greater_than = False
                    temp.append(current)
            elif len(temp) == 0:
                temp.append(dataframe.iloc[i, 0])

    return np.array(X), np.array(y)


def which_maneuver(data, maneuver, I, P):
    X, y = None, None

    if maneuver == 'BREAKING':
        X, y = breaking_feature_extraction(data, I, P)
    elif maneuver == 'ACCELERATING':
        X, y = acceleration_feature_extraction(data, I, P)

    return X, y


run_matlab_in_python()
acceleration, speed = excel_to_python()

acceleration_without_time = acceleration.drop(columns='Timestamp')
speed_without_time = speed.drop(columns='Timestamp')

accelration_filtered = acceleration_without_time.apply(lambda x: savgol_filter(x, window_length=5, polyorder=2))

acceleration_standarised = standarisation(accelration_filtered)
acceleration_normalised = normalisation(accelration_filtered)

# plot(acceleration_without_time, acceleration_standarised, 'STANDARISED')
# plot(acceleration_without_time, acceleration_normalised, 'NORMALISED')

X_breaking, y_breaking = which_maneuver(acceleration_standarised, 'BREAKING', I=0.1, P=1)
X_accelerating, y_accelerating = which_maneuver(acceleration_standarised, 'ACCELERATING', I=0.1, P=1)

print("ALL MANEUVERS: {}".format(len(y_breaking) + len(y_accelerating)))
print(40 * "=")
aggresive_breaking = 0
normal_breaking = 0

for i in range(len(y_breaking)):
    if y_breaking[i] == 0:
        normal_breaking += 1
    else:
        aggresive_breaking += 1

print("ALL BRAKING MANEUVERS: {}\n"
      "CASES OF AGGRESIVE BREAKING: {}\n"
      "CASES OF NORMAL BREAKING: {}".format(len(y_breaking), aggresive_breaking, normal_breaking))
print(40 * '=')

aggresive_accelerating = 0
normal_accelerating = 0

for i in range(len(y_accelerating)):
    if y_accelerating[i] == 0:
        normal_accelerating += 1
    else:
        aggresive_accelerating += 1

print("ALL ACCELERATING MANEUVERS: {}\n"
      "CASES OF AGGRESIVE ACCELERATING: {}\n"
      "CASES OF NORMAL ACCELERATING: {}".format(len(y_accelerating), aggresive_accelerating, normal_accelerating))
print(40 * '=')

avg_speed = speed['speed'].mean()
max_speed = speed['speed'].max()

print('Average speed: {} m/s'.format(round(avg_speed, 2)))
print('Maximum speed: {} m/s'.format(round(max_speed, 2)))

print(40 * '=')

avg_acc = acceleration['Y'].mean()
max_acc = acceleration['Y'].max()

print('Average acceleration: {} m/s^2'.format(round(avg_acc, 2)))
print('Maximum acceleration: {} m/s^2'.format(round(max_acc, 2)))

print(40 * '=')

date = datetime.datetime.strptime(acceleration['Timestamp'].iat[-1], "%d-%b-%Y %H:%M:%S.%f").date()
start_time = datetime.datetime.strptime(acceleration['Timestamp'].iat[0], "%d-%b-%Y %H:%M:%S.%f").time()
end_time = datetime.datetime.strptime(acceleration['Timestamp'].iat[-1], "%d-%b-%Y %H:%M:%S.%f").time()
time_of_drive = datetime.datetime.strptime(acceleration['Timestamp'].iat[-1],
                                           "%d-%b-%Y %H:%M:%S.%f") - datetime.datetime.strptime(
    acceleration['Timestamp'].iat[0], "%d-%b-%Y %H:%M:%S.%f")

print('Date: {}'.format(date))
print('Start time: {}'.format(start_time))
print('End time: {}'.format(end_time))
print('The time of the drive is: {}'.format(time_of_drive))


def send_data():
    all_maneuvers = len(y_breaking) + len(y_accelerating)
    breaking_len = len(y_breaking)
    acceleration_len = len(y_accelerating)
    percent_of_aggresive_breakins = aggresive_breaking / len(y_breaking)
    percent_of_normal_breakings = normal_breaking / len(y_breaking)
    percent_of_aggresive_accelerating = aggresive_accelerating / len(y_accelerating)
    percent_of_normal_accelerating = normal_accelerating / len(y_accelerating)

    return date, start_time, \
           end_time, time_of_drive, \
           round(avg_speed*18/5, 2), round(max_speed*18/5, 2), \
           round(avg_acc, 2), round(max_acc, 2), \
           all_maneuvers, breaking_len, \
           acceleration_len, normal_breaking, \
           aggresive_breaking, normal_accelerating, \
           aggresive_accelerating, round(percent_of_aggresive_breakins, 2), \
           round(percent_of_normal_breakings, 2), round(percent_of_aggresive_accelerating, 2), \
           round(percent_of_normal_accelerating, 2)
