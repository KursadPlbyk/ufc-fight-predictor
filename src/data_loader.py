import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_historical_data(file_path):
    """
    Load historical fight data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing historical fight data.

    Returns:
    - pd.DataFrame: A DataFrame containing the historical fight data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def clean_data(raw_df):
    """
    Clean the raw historical fight data.

    Parameters:
    - raw_df (pd.DataFrame): The raw DataFrame containing historical fight data.

    Returns:
    - cleaned_df (pd.DataFrame): A cleaned DataFrame with relevant columns and no missing values.
    """
    df = raw_df.copy()

    # drop irrelevant columns (in-fight/round-level stats = data leakage, not known before a fight happens)
    round_markers = ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']
    columns_to_drop = [col for col in df.columns if any(marker in col for marker in round_markers)]

    columns_to_drop += ['event_city', 'event_state', 'event_country', 'f_1_fighter_nickname','f_2_fighter_nickname']

    for i in (1, 2):
        columns_to_drop += [
            f'f_{i}_knockdowns',
            f'f_{i}_total_strikes_att',
            f'f_{i}_total_strikes_succ',
            f'f_{i}_sig_strikes_att',
            f'f_{i}_sig_strikes_succ',
            f'f_{i}_takedown_att',
            f'f_{i}_takedown_succ',
            f'f_{i}_submission_att',
            f'f_{i}_reversals',
            f'f_{i}_ctrl_time_sec',
            f'f_{i}_odds',
            f'f_{i}_ko_odds',
            f'f_{i}_sub_odds',
        ]
        df[f'f_{i}_ranking'] = df[f'f_{i}_ranking'].fillna("Unranked")
        df[f'f_{i}_fighter_nc_dq'] = df[f'f_{i}_fighter_nc_dq'].fillna(0)
        df[f'f_{i}_fighter_reach_cm'] = df[f'f_{i}_fighter_reach_cm'].fillna(df[f'f_{i}_fighter_reach_cm'].median())



    df = df.drop(columns=columns_to_drop)
    df = df.drop_duplicates()  # Drop duplicate rows
    df = df.dropna()  # Drop rows with any missing values
    

    return df

if __name__ == "__main__":
    file_path = "data/raw_historical_fights.csv"
    historical_data = load_historical_data(file_path)
    if historical_data is not None:
        print("Historical data loaded successfully.\n")
        print(f"Data shape: {historical_data.shape}\n")
        historical_data.info()  # Display DataFrame info
        print(f"Missing values per column:\n{historical_data.isnull().sum().sort_values(ascending=False)}\n") # Display count of missing values in each column    
        print(f"missing values in percentage:\n{(historical_data.isnull().mean() * 100).sort_values(ascending=False)}\n") # Display percentage of missing values in each column

        print("--------------------------------------------------------------------\n")

        print("\nCleaning data...\n")
        cleaned_data = clean_data(historical_data)
        print("Data cleaned successfully.\n")
        print(f"Cleaned data shape: {cleaned_data.shape}\n")
        cleaned_data_csv = cleaned_data.to_csv("data/cleaned_fights.csv", index=False)
        print(f"Missing values per column:\n{cleaned_data.isnull().sum().sort_values(ascending=False)}\n") # Display count of missing values in each column    
        print(f"missing values in percentage:\n{(cleaned_data.isnull().mean() * 100).sort_values(ascending=False)}")
        