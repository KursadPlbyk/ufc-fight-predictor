from src.scraper import get_fighter_current_stats
import re
import glob
from datetime import datetime
import joblib
import pandas as pd

# 95th-percentile absolute diffs seen in the historical training data (data/features.csv).
# Fights matched within the same weight class rarely show bigger gaps than this - a live
# matchup that exceeds them is likely a cross-weight-class "fantasy fight" the model has
# very few (or no) comparable training examples for, so predictions should be flagged.
HEIGHT_DIFF_WARNING_CM = 12.7
WEIGHT_DIFF_WARNING_LBS = 25

# Human-readable labels for the model's feature names, used when printing a report.
FEATURE_LABELS = {
    'total_fights_f1': 'Total Fights (Fighter A)',
    'total_fights_f2': 'Total Fights (Fighter B)',
    'experience_diff': 'Experience Difference (Total Fights)',
    'win_ratio_f1': 'Win Ratio (Fighter A)',
    'win_ratio_f2': 'Win Ratio (Fighter B)',
    'win_ratio_diff': 'Win Ratio Difference',
    'slpm_diff': 'Significant Strikes Landed per Minute Difference',
    'str_acc_diff': 'Striking Accuracy Difference',
    'sapm_diff': 'Strikes Absorbed Advantage (fewer absorbed = better)',
    'str_def_diff': 'Striking Defense Difference',
    'td_avg_diff': 'Takedown Average Difference',
    'td_acc_diff': 'Takedown Accuracy Difference',
    'td_def_diff': 'Takedown Defense Difference',
    'sub_avg_diff': 'Submission Average Difference',
    'reach_diff': 'Reach Difference (cm)',
    'height_diff': 'Height Difference (cm)',
    'weight_diff': 'Weight Difference (lbs)',
    'age_f1': 'Age (Fighter A)',
    'age_f2': 'Age (Fighter B)',
    'age_diff': 'Age Advantage (younger fighter = better)',
}

# Features where a HIGHER raw value is actually worse for Fighter A (confirmed via
# correlation with 'target' in data/features.csv: both correlate negatively). Their
# sign is flipped only for report display, so "positive = advantage for Fighter A"
# holds consistently everywhere in the report - the raw, unflipped values are still
# what gets fed into the model.
INVERTED_DISPLAY_FEATURES = {'sapm_diff', 'age_diff'}

def normalize_fighter_stats(raw_stats):
    """
    Normalize the raw stats scraped from ufcstats.com into a consistent format for feature engineering.
    
    Parameters:
    - raw_stats (dict): A dictionary containing the raw stats of a fighter.

    Returns:
    - dict: A dictionary containing the normalized stats of a fighter.
    """

    normalized_stats = {}
    teile = raw_stats['Height'].split("'")
    if len(teile) == 2:
        feet = int(teile[0])
        inches = int(teile[1].replace('"', '').strip())
        normalized_stats['Height_cm'] = feet * 30.48 + inches * 2.54  # Convert to centimeters
    else:
        feet = int(teile[0])
        normalized_stats['Height_cm'] = feet * 30.48  # Convert to centimeters

    normalized_stats['Weight_lbs'] = float(raw_stats['Weight'].replace(' lbs', '').strip())

    if raw_stats['Reach'] == '--':
        normalized_stats['Reach_cm'] = None
    else:
        normalized_stats['Reach_cm'] = float(raw_stats['Reach'].replace('"', '').strip()) * 2.54

    normalized_stats['Str_Acc_dec'] = float(raw_stats['Str. Acc.'].replace('%', '').strip()) / 100.0
    normalized_stats['Str_Def_dec'] = float(raw_stats['Str. Def'].replace('%', '').strip()) / 100.0
    normalized_stats['TD_Acc_dec'] = float(raw_stats['TD Acc.'].replace('%', '').strip()) / 100.0
    normalized_stats['TD_Def_dec'] = float(raw_stats['TD Def.'].replace('%', '').strip()) / 100.0

    normalized_stats['SlpM'] = float(raw_stats['SLpM'])
    normalized_stats['SApM'] = float(raw_stats['SApM'])
    normalized_stats['TD_Avg'] = float(raw_stats['TD Avg.'])
    normalized_stats['Sub_Avg'] = float(raw_stats['Sub. Avg.'])

    normalized_stats['DOB_date'] = datetime.strptime(raw_stats['DOB'], '%b %d, %Y')
    normalized_stats['Wins'] = raw_stats['wins']
    normalized_stats['Losses'] = raw_stats['losses']
    normalized_stats['Draws'] = raw_stats['draws']

    return normalized_stats

def build_feature_row(stats_f1, stats_f2):
    """
    Build a feature row for the model based on the normalized stats of two fighters.

    Parameters:
    - stats_f1 (dict): A dictionary containing the normalized stats of fighter 1.
    - stats_f2 (dict): A dictionary containing the normalized stats of fighter 2.

    Returns:
    - pd.DataFrame: A DataFrame containing a single row of features for the model.
    """

    total_fights_f1 = stats_f1['Wins'] + stats_f1['Losses'] + stats_f1['Draws']
    total_fights_f2 = stats_f2['Wins'] + stats_f2['Losses'] + stats_f2['Draws']
    experience_diff = total_fights_f1 - total_fights_f2
    win_ratio_f1 = stats_f1['Wins'] / total_fights_f1 if total_fights_f1 > 0 else 0
    win_ratio_f2 = stats_f2['Wins'] / total_fights_f2 if total_fights_f2 > 0 else 0
    win_ratio_diff = win_ratio_f1 - win_ratio_f2
    slpm_diff = stats_f1['SlpM'] - stats_f2['SlpM']
    str_acc_diff = stats_f1['Str_Acc_dec'] - stats_f2['Str_Acc_dec']
    sapm_diff = stats_f1['SApM'] - stats_f2['SApM']
    str_def_diff = stats_f1['Str_Def_dec'] - stats_f2['Str_Def_dec']
    td_avg_diff = stats_f1['TD_Avg'] - stats_f2['TD_Avg']
    td_acc_diff = stats_f1['TD_Acc_dec'] - stats_f2['TD_Acc_dec']
    td_def_diff = stats_f1['TD_Def_dec'] - stats_f2['TD_Def_dec']
    sub_avg_diff = stats_f1['Sub_Avg'] - stats_f2['Sub_Avg']
    if stats_f1['Reach_cm'] is None or stats_f2['Reach_cm'] is None:
        reach_diff = 0  # unknown reach for one fighter - assume no advantage either way
    else:
        reach_diff = stats_f1['Reach_cm'] - stats_f2['Reach_cm']
    height_diff = stats_f1['Height_cm'] - stats_f2['Height_cm']
    weight_diff = stats_f1['Weight_lbs'] - stats_f2['Weight_lbs']

    today = datetime.now()
    age_f1 = (today - stats_f1['DOB_date']).days / 365.25
    age_f2 = (today - stats_f2['DOB_date']).days / 365.25
    age_diff = age_f1 - age_f2

    feature_row = pd.DataFrame([{
        'total_fights_f1': total_fights_f1,
        'total_fights_f2': total_fights_f2,
        'experience_diff': experience_diff,
        'win_ratio_f1': win_ratio_f1,
        'win_ratio_f2': win_ratio_f2,
        'win_ratio_diff': win_ratio_diff,
        'slpm_diff': slpm_diff,
        'str_acc_diff': str_acc_diff,
        'sapm_diff': sapm_diff,
        'str_def_diff': str_def_diff,
        'td_avg_diff': td_avg_diff,
        'td_acc_diff': td_acc_diff,
        'td_def_diff': td_def_diff,
        'sub_avg_diff': sub_avg_diff,
        'reach_diff': reach_diff,
        'height_diff': height_diff,
        'weight_diff': weight_diff,
        'age_f1': age_f1,
        'age_f2': age_f2,
        'age_diff': age_diff,
    }])

    return feature_row

def load_trained_model():
    """
    Load the trained machine learning model from a file.

    Returns:
    - model: The loaded machine learning model.
    """
    found_model_files = glob.glob('models/ufc_predictor_*_model.pkl')
    if not found_model_files:
        raise FileNotFoundError("No trained model file found in the 'models' directory.")
    print(f"Loading model from {found_model_files[0]}...\n")
    model = joblib.load(found_model_files[0])
    return model

def predict_fight_outcome(fighter_f1_name, fighter_f2_name):
    """
    Predict the outcome of a fight between two fighters.

    Parameters:
    - fighter_f1_name (str): The full name of fighter A.
    - fighter_f2_name (str): The full name of fighter B.

    Returns:
    - dict: A dictionary containing the predicted probabilities of each fighter winning.
    """

    raw_f1 = get_fighter_current_stats(fighter_f1_name)
    raw_f2 = get_fighter_current_stats(fighter_f2_name)

    stats_f1 = normalize_fighter_stats(raw_f1)
    stats_f2 = normalize_fighter_stats(raw_f2)

    feature_row = build_feature_row(stats_f1, stats_f2)

    model = load_trained_model()
    prediction = model.predict(feature_row)
    probabilities = model.predict_proba(feature_row)[0]

    result = {
    'fighter_a': fighter_f1_name,
    'fighter_b': fighter_f2_name,
    'predicted_winner': fighter_f1_name if prediction[0] == 1 else fighter_f2_name,
    'confidence': max(probabilities),
    'fighter_a_probability': probabilities[1],
    'fighter_b_probability': probabilities[0],
    'height_diff': feature_row['height_diff'].iloc[0],
    'weight_diff': feature_row['weight_diff'].iloc[0],
    'features': feature_row.iloc[0].to_dict(),
    }

    return result


def generate_report(result):
    """
    Build a human-readable text report from a predict_fight_outcome() result.

    Parameters:
    - result (dict): The dictionary returned by predict_fight_outcome().

    Returns:
    - str: A formatted, multi-line report.
    """

    lines = [
        f"{result['fighter_a']} vs {result['fighter_b']}",
        "",
        "Features used for this prediction: \n",
        f"(A positive value favors {result['fighter_a']}, a negative value favors {result['fighter_b']})\n",
    ]
    for feature_name, feature_value in result['features'].items():
        label = FEATURE_LABELS.get(feature_name, feature_name)
        display_value = -feature_value if feature_name in INVERTED_DISPLAY_FEATURES else feature_value
        lines.append(f"  {label}: {display_value:.2f}")

    lines += [
        "",
        f"Predicted winner: {result['predicted_winner']} ({result['confidence']:.1%} confidence)",
        f"{result['fighter_a']}: {result['fighter_a_probability']:.1%}",
        f"{result['fighter_b']}: {result['fighter_b_probability']:.1%}",
    ]

    if abs(result['height_diff']) > HEIGHT_DIFF_WARNING_CM or abs(result['weight_diff']) > WEIGHT_DIFF_WARNING_LBS:
        lines.append(
            "\nWarning: unusually large height/weight difference for this matchup "
            "(bigger than 95% of the historical same-weight-class fights the model was "
            "trained on). This looks like a cross-weight-class 'fantasy fight' - treat "
            "the prediction with caution."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    fighter_a = "Joshua Van"
    fighter_b = "Alexandre Pantoja"

    print(f"Scraping current stats for {fighter_a}...\n")
    raw_a = get_fighter_current_stats(fighter_a)
    print("Raw stats:\n", raw_a)
    print("\nNormalizing stats...\n")
    normalized_a = normalize_fighter_stats(raw_a)
    print("Normalized stats:\n", normalized_a)

    print(f"\n---------------------------------------------------\n")

    print(f"Scraping current stats for {fighter_b}...\n")
    raw_b = get_fighter_current_stats(fighter_b)
    print("Raw stats:\n", raw_b)
    print("\nNormalizing stats...\n")
    normalized_b = normalize_fighter_stats(raw_b)
    print("Normalized stats:\n", normalized_b)

    print(f"\n---------------------------------------------------\n")

    print(f"Building feature row for {fighter_a} vs {fighter_b}...\n")
    feature_row = build_feature_row(normalized_a, normalized_b)
    print(feature_row.to_string())

    print(f"\n---------------------------------------------------\n")

    print(f"Predicting outcome for {fighter_a} vs {fighter_b}...\n")
    result = predict_fight_outcome(fighter_a, fighter_b)

    print(f"\n---------------------------------------------------\n")

    print(generate_report(result))
    