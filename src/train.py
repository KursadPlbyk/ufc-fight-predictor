import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def load_features(file_path):
    """
    Load engineered features from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing engineered features.

    Returns:
    - pd.DataFrame: A DataFrame containing the engineered features.
    """
    try:
        features = pd.read_csv(file_path)
        X = features[['total_fights_f1', 'total_fights_f2', 'experience_diff', 'win_ratio_f1', 'win_ratio_f2',
                     'win_ratio_diff', 'slpm_diff', 'str_acc_diff', 'sapm_diff', 'str_def_diff', 'td_avg_diff'
                     , 'td_acc_diff', 'td_def_diff', 'sub_avg_diff', 'reach_diff', 'height_diff', 'weight_diff',
                     'age_f1', 'age_f2', 'age_diff']]

        y = features['target']

        return X, y
    except Exception as e:
        print(f"Error loading features from {file_path}: {e}")
        return None, None


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - X (pd.DataFrame): The feature set.
    - y (pd.Series): The target variable.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train, X_val, y_val):
    """
    Train a machine learning model on the training data.

    Parameters:
    - X_train (pd.DataFrame): The training feature set.
    - y_train (pd.Series): The training target variable.
    - X_val (pd.DataFrame): Validation feature set, used for XGBoost early stopping.
    - y_val (pd.Series): Validation target variable, used for XGBoost early stopping.

    Returns:
    - model1 (RandomForestClassifier), model2 (XGBClassifier): The trained machine learning models.
    """
    # Initialize the model (you can choose any classifier you prefer)
    model1 = RandomForestClassifier(n_estimators=200, random_state=42)
    model1.fit(X_train, y_train)

    # n_estimators=1000 is just a generous ceiling - early stopping decides the real count
    model2 = XGBClassifier(
        n_estimators=1000,
        eval_metric='logloss',
        early_stopping_rounds=20,
        random_state=42,
    )
    model2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model1, model2


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Parameters:
    - model: The trained machine learning model.
    - X_test (pd.DataFrame): The test feature set.
    - y_test (pd.Series): The test target variable.

    Returns:
    - metrics (dict): A dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.figure_.savefig(f'models/plots/confusion_matrix_{model.__class__.__name__}.png')
    plt.close(disp.figure_)

    diagram_plot = pd.Series(model.feature_importances_, index=X_test.columns).plot(kind='barh', figsize=(10, 8))
    diagram_plot.figure.tight_layout()
    diagram_plot.figure.savefig(f'models/plots/feature_importance_{model.__class__.__name__}.png')
    plt.close(diagram_plot.figure)

    metrics = {
        'accuracy': acc_score,
        'precision': prec_score,
        'recall': rec_score,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics
