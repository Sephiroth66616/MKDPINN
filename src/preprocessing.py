import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split,TensorDataset
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0,
                                                                                                        drop=True)

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements - (sequence_length - 1)), range(sequence_length, num_elements + 1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    data_gen = (list(gen_train_data(df[df['unit_nr'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length - 1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    label_gen = [gen_labels(df[df['unit_nr'] == unit_nr], sequence_length, label)
                 for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def get_data(dataset, sensors, sequence_length, alpha, threshold):
    # files
    dir_path = './data/'
    train_file = 'train_' + dataset + '.txt'
    test_file = 'test_' + dataset + '.txt'
    # columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
    # data readout
    train = pd.read_csv((dir_path + train_file), sep=r'\s+', header=None,
                        names=col_names)
    test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None,
                       names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_' + dataset + '.txt'), sep=r'\s+', header=None,
                         names=['RemainingUsefulLife'])

    # create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)

        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

    # create sequences for test
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)
    # clip y_test to threshold
    y_test['RemainingUsefulLife'] = y_test['RemainingUsefulLife'].clip(upper=threshold)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']


def get_data_loader(dataset, sensors, sequence_length, alpha, threshold, batch_size,train_size=0.9,random_state=3407):
    dir_path = './data/'
    train_file = 'train_' + dataset + '.txt'
    test_file = 'test_' + dataset + '.txt'
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i + 1) for i in range(21)]
    col_names = index_names + setting_names + sensor_names
    train = pd.read_csv((dir_path + train_file), sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_' + dataset + '.txt'), sep=r'\s+', header=None, names=['RemainingUsefulLife'])
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)
    drop_sensors = [element for element in sensor_names if element not in sensors]
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)
        t_train = gen_label_wrapper(X_train_pre, sequence_length, ['time_cycles'], train_unit)

        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)
        t_val = gen_label_wrapper(X_train_pre, sequence_length, ['time_cycles'], val_unit)

    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)
    y_test['RemainingUsefulLife'] = y_test['RemainingUsefulLife'].clip(upper=threshold)

    t_test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], 1, ['time_cycles'], -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    t_test = np.concatenate(list(t_test_gen)).astype(np.float32)
    t_test = t_test.reshape(-1,1)


    x_train = torch.tensor(x_train).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)
    t_train = torch.tensor(t_train).to(torch.float32)

    x_val = torch.tensor(x_val).to(torch.float32)
    y_val = torch.tensor(y_val).to(torch.float32)
    t_val = torch.tensor(t_val).to(torch.float32)

    x_test = torch.tensor(x_test).to(torch.float32)
    y_test_rul = torch.tensor(y_test['RemainingUsefulLife']).to(torch.float32)
    t_test = torch.tensor(t_test).to(torch.float32)

    train_dataset = TensorDataset(x_train,t_train, y_train.squeeze(-1))
    val_dataset = TensorDataset(x_val,t_val, y_val.squeeze(-1))
    test_dataset = TensorDataset(x_test, t_test,y_test_rul)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)

    return train_loader, val_loader, test_loader