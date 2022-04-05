from glob import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


def calc_mse_loss(df, category_grouped_df):
    """TODO: write docstring"""
    grouped_df = df.groupby("class_id").count()[["fname"]] / len(df) * 100
    df_temp = category_grouped_df.join(
        grouped_df, on="class_id", how="left", lsuffix="_main")
    df_temp.fillna(0, inplace=True)
    df_temp["diff"] = (df_temp["fname"] - df_temp["fname"])**2
    mse_loss = np.mean(df_temp["diff"])
    return mse_loss


def calc_mse_loss_diff(df, batch_df, category_grouped_df):
    """TODO: write docstring"""
    return (
        calc_mse_loss(df, category_grouped_df) - 
        calc_mse_loss(df.append(batch_df, ignore_index=True), category_grouped_df))
    

def calculate_loss(pct, df_len, total_len, mse_loss_diff, weight=.01):
    """TODO: write docstring"""
    len_diff = pct - (df_len / total_len)
    len_loss_diff = len_diff * abs(len_diff)
    loss = (weight * mse_loss_diff) + ((1-weight) * len_loss_diff)
    return loss


def write_file(df, dest):
    """TODO: write docstring"""
    fnames = df["fname"].unique()
    with open(dest, "w") as f:
        for fname in fnames:
            f.write(f"{fname}\n")


def print_summary(df, set_name, total):
    """TODO: write docstring"""
    print(f"""
        {set_name} set:
        Number of files {len(df['fname'].unique())}
        Number of labels {len(df)}\n
        Number of labels {round(100*(len(df) / total), 0)}\n
        Labels Distribution\n{df['class_id'].value_counts()}\n
    """)
    
    
def create_labels_df(folder):
    """TODO: write docstring"""
    files = glob(f"{folder}*.txt")
    fnames, class_ids = [], []
    for fname in files:
        with open(fname, "r") as f:
            for line in f.readlines():
                class_id = int(line.split(" ")[0])
                class_ids.append(class_id)
                fnames.append(fname.split("/")[-1])
    df = pd.DataFrame({"fname": fnames, "class_id": class_ids})
    df = df.reindex(np.random.permutation(df.index))
    return df


def yolo_obj_detection_split_data(folder, train_pct=.7, test_pct=.2, val_pct=.1, weight=.01, batch_size=1):
    """TODO: write docstring"""
    df = create_labels_df(folder)  # Create master dataframe with a line per label
    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Create dataframes for split
    subject_grouped_df = df.groupby(["fname"], sort=False, as_index=False)  # Group by file
    category_grouped_df = df.groupby("class_id").count()[["fname"]] / len(df) * 100  # Get class percentage distribution
    print(f"{len(df['fnames'].unique())} number of images")
    print(f"{len(df['class_id'])} number of labels")
    print("\nClass Distribution\n", df["class_id"].value_counts())
    print("\nClass Percentage Distribution\n", category_grouped_df)
    batch_df = df[0:0]  # create empty df
    for i, (_, group) in enumerate(subject_grouped_df):  # iterate over file groups
        # Populate empty dataframes with an instance each to start
        if i == 0:
            df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
            continue
        elif i == 1:
            df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
            continue
        elif i == 2:
            df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
            continue

        batch_df = batch_df.append(group)
        calc_mse_loss_diff(df_train, batch_df, category_grouped_df)
        mse_loss_diff_train = calc_mse_loss_diff(df_train, batch_df, category_grouped_df)
        mse_loss_diff_val = calc_mse_loss_diff(df_val, batch_df, category_grouped_df)
        mse_loss_diff_test = calc_mse_loss_diff(df_test, batch_df, category_grouped_df)

        total_records = len(df_train) + len(df_val) + len(df_test)
        loss_train = calculate_loss(
            train_pct, len(df_train), total_records, mse_loss_diff_train)
        loss_val = calculate_loss(
            val_pct, len(df_val), total_records, mse_loss_diff_val)
        loss_test = calculate_loss(
            test_pct, len(df_test), total_records, mse_loss_diff_test)

        if (max(loss_train, loss_val, loss_test) == loss_train):
            df_train = df_train.append(batch_df, ignore_index=True)
        elif (max(loss_train, loss_val, loss_test) == loss_val):
            df_val = df_val.append(batch_df, ignore_index=True)
        else:
            df_test = df_test.append(batch_df, ignore_index=True)
        batch_df = df[0:0]
    print_summary(df_train, "Training", total_records)
    print_summary(df_val, "Validation", total_records)
    print_summary(df_test, "Test", total_records)
    write_file(df_train, f"{folder}train.txt")
    write_file(df_val, f"{folder}val.txt")
    write_file(df_test, f"{folder}test.txt")
