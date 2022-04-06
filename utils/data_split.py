from glob import glob
import os
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


def write_file(df, folder, set_name):
    """TODO: write docstring"""
    fnames = df["fname"].str.replace("txt", "jpg").unique()
    with open(f"{folder}model_files/{set_name}", "w") as f:
        for fname in fnames:
            f.write(f"{folder}data/{fname}\n")


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


def create_model_files(folder, labels, df_train, df_val, df_test):
    # Write dataset files
    write_file(df_train, folder, "train.txt")
    write_file(df_val, folder, "val.txt")
    write_file(df_test, folder, "test.txt")
    # Create obj.names file i.e., label names
    with open(f"{folder}model_files/obj.names", "w") as f:
        f.write('\n'.join(labels))
    # Create obj.data files i.e., info for training
    with open(f"{folder}model_files/obj.data", 'w') as f:
        f.write(f"classes = {len(labels)}\n")
        f.write(f"train = {folder}model_files/train.txt\n")
        f.write(f"valid = {folder}model_files/val.txt\n")
        f.write(f"names = {folder}model_files/obj.names\n")
        f.write(f"backup = backup/")

    with open(f"{folder}model_files/obj_test.data", 'w') as f:
        f.write(f"classes = {len(labels)}\n")
        f.write(f"train = {folder}model_files/train.txt\n")
        f.write(f"valid = {folder}model_files/test.txt\n")
        f.write(f"names = {folder}model_files/obj.names\n")
        f.write(f"backup = backup/")


def yolo_obj_detection_setup(folder, labels, train_pct=.7, test_pct=.2, val_pct=.1):
    """TODO: write docstring"""
    assert train_pct + test_pct + val_pct == 1., "Percentages must add up to 1"
    df = create_labels_df(f"{folder}data/")  # Create master dataframe with a line per label
    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Create dataframes for split
    subject_grouped_df = df.groupby(["fname"], sort=False, as_index=False)  # Group by file
    category_grouped_df = df.groupby("class_id").count()[["fname"]] / len(df) * 100  # Get class percentage distribution
    print(f"{len(df['fname'].unique())} number of images")
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
    create_model_files(folder, labels, df_train, df_val, df_test)

