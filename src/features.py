import pandas as pd
import numpy as np
from src.data_loader import load_historical_data, clean_data

def debias_fighter_order(df, seed=42):
    """
    Randomly swap the f_1/f_2 assignment for ~50% of rows.

    In the raw data, f_1 is the fight winner 100% of the time (the scraper
    lists the winner first). Without this, 'target' would be constant (always 1)
    and every f_1-f_2 diff feature would secretly encode the outcome by position
    instead of by actual stat differences.

    Parameters:
    - df (pd.DataFrame): Cleaned fight data, still with the original (biased) f_1/f_2 order.
    - seed (int): Random seed, so the swap is reproducible across runs.

    Returns:
    - pd.DataFrame: A copy of df with ~50% of rows' f_1_*/f_2_* columns swapped.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    swap_mask = rng.random(len(df)) < 0.5

    f1_columns = [col for col in df.columns if col.startswith('f_1_')]
    for f1_col in f1_columns:
        f2_col = 'f_2_' + f1_col[len('f_1_'):]
        if f2_col in df.columns:
            f1_values = df.loc[swap_mask, f1_col].copy()
            df.loc[swap_mask, f1_col] = df.loc[swap_mask, f2_col]
            df.loc[swap_mask, f2_col] = f1_values

    return df

def engineer_features(file_path):
    """
    Load and clean historical fight data, then engineer features for modeling.

    Parameters:
    - file_path (str): The path to the CSV file containing historical fight data.

    Returns:
    - pd.DataFrame: A DataFrame containing engineered features for modeling.
    """
    # Load historical data
    raw_data = load_historical_data(file_path)
    
    if raw_data is None:
        print("Failed to load data. Exiting feature engineering.")
        return None

    # Clean the data
    cleaned_data = clean_data(raw_data)

    # De-bias fighter order (f_1 is the winner 100% of the time otherwise)
    cleaned_data = debias_fighter_order(cleaned_data)

    event_date = pd.to_datetime(cleaned_data['event_date'])
    f1_dob = pd.to_datetime(cleaned_data['f_1_fighter_dob'])
    f2_dob = pd.to_datetime(cleaned_data['f_2_fighter_dob'])

    age_f1 = (event_date - f1_dob).dt.days / 365.25
    age_f2 = (event_date - f2_dob).dt.days / 365.25
    age_diff = age_f1 - age_f2

    total_fights_f1 = cleaned_data['f_1_fighter_w'] + cleaned_data['f_1_fighter_l'] + cleaned_data['f_1_fighter_d']
    total_fights_f2 = cleaned_data['f_2_fighter_w'] + cleaned_data['f_2_fighter_l'] + cleaned_data['f_2_fighter_d']

    win_ratio_f1 = cleaned_data['f_1_fighter_w'] / total_fights_f1.replace(0, 1)
    win_ratio_f2 = cleaned_data['f_2_fighter_w'] / total_fights_f2.replace(0, 1)
    win_ratio_diff = win_ratio_f1 - win_ratio_f2

    slpm_diff = (cleaned_data['f_1_fighter_SlpM'] - cleaned_data['f_2_fighter_SlpM'])

    str_acc_diff = (cleaned_data['f_1_fighter_Str_Acc'] - cleaned_data['f_2_fighter_Str_Acc'])

    sapm_diff = (cleaned_data['f_1_fighter_SApM'] - cleaned_data['f_2_fighter_SApM'])

    str_def_diff = (cleaned_data['f_1_fighter_Str_Def'] - cleaned_data['f_2_fighter_Str_Def'])

    td_avg_diff = (cleaned_data['f_1_fighter_TD_Avg'] - cleaned_data['f_2_fighter_TD_Avg'])


    td_acc_diff = (cleaned_data['f_1_fighter_TD_Acc'] - cleaned_data['f_2_fighter_TD_Acc'])

    td_def_diff = (cleaned_data['f_1_fighter_TD_Def'] - cleaned_data['f_2_fighter_TD_Def'])

    sub_avg_diff = (cleaned_data['f_1_fighter_Sub_Avg'] - cleaned_data['f_2_fighter_Sub_Avg'])

    reach_diff = (cleaned_data['f_1_fighter_reach_cm'] - cleaned_data['f_2_fighter_reach_cm'])

    height_diff = (cleaned_data['f_1_fighter_height_cm'] - cleaned_data['f_2_fighter_height_cm'])

    weight_diff = (cleaned_data['f_1_fighter_weight_lbs'] - cleaned_data['f_2_fighter_weight_lbs'])


    cleaned_data['total_fights_f1'] = total_fights_f1
    cleaned_data['total_fights_f2'] = total_fights_f2
    cleaned_data['win_ratio_f1'] = win_ratio_f1
    cleaned_data['win_ratio_f2'] = win_ratio_f2
    cleaned_data['win_ratio_diff'] = win_ratio_diff
    cleaned_data['experience_diff'] = total_fights_f1 - total_fights_f2
    cleaned_data['slpm_diff'] = slpm_diff
    cleaned_data['str_acc_diff'] = str_acc_diff
    cleaned_data['sapm_diff'] = sapm_diff
    cleaned_data['str_def_diff'] = str_def_diff
    cleaned_data['td_avg_diff'] = td_avg_diff
    cleaned_data['td_acc_diff'] = td_acc_diff
    cleaned_data['td_def_diff'] = td_def_diff
    cleaned_data['sub_avg_diff'] = sub_avg_diff
    cleaned_data['reach_diff'] = reach_diff
    cleaned_data['height_diff'] = height_diff
    cleaned_data['weight_diff'] = weight_diff
    cleaned_data['age_f1'] = age_f1
    cleaned_data['age_f2'] = age_f2
    cleaned_data['age_diff'] = age_diff
    cleaned_data['target'] = (cleaned_data['winner'] == cleaned_data['f_1_name']).astype(int)


    return cleaned_data

if __name__ == "__main__":
    file_path = "data/raw_historical_fights.csv"
    features_data = engineer_features(file_path)
    if features_data is not None:
        print(f"Features shape: {features_data.shape}\n")
        print(f"Target distribution:\n{features_data['target'].value_counts()}\n")
        features_data.to_csv("data/features.csv", index=False)
        print("Features saved to data/features.csv")
        