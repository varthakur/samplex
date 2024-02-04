  # To handle the data
import pandas as pd

# Sampling
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.under_sampling import TomekLinks as TL
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss as NM

# ML Models
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
import xgboost as xgb

# To measure accuracy/recall
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import recall_score

# Load data
credit_data = pd.read_csv('/content/Creditcard_data.csv')

# Check for unbalanced column - 'Class'
class_ratio = sum(credit_data['Class'] == 1) / (sum(credit_data['Class'] == 0) + sum(credit_data['Class'] == 1))

def build_model(model_type, train_features, train_target):
    if model_type == 'LogisticRegression':
        model = LR(max_iter=1000, solver='newton-cg')
    elif model_type == 'RandomForestClassifier':
        model = RFC()
    elif model_type == 'SVC':
        model = SVC()
    elif model_type == 'KNeighborsClassifier':
        model = KNC()
    elif model_type == 'XGBClassifier':
        model = xgb.XGBClassifier()
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    model.fit(train_features, train_target)
    return model

def calculate_performance(metric, true_labels, predicted_labels):
    if metric == 'recall':
        score = recall_score(true_labels, predicted_labels)
    elif metric == 'accuracy':
        score = accuracy_score(true_labels, predicted_labels)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return score

# Creating columns for the resultant data frame
result_dataframe = pd.DataFrame()

# For reproducibility
seed_value = 42 

# Defining a function to evaluate the accuracy of a model based on five different sampling techniques
def evaluate_models(model_list, metric):
    sampling_1_scores = []
    sampling_2_scores = []
    sampling_3_scores = []
    sampling_4_scores = []
    sampling_5_scores = []

    for model_type in model_list:
        # A) Random Under-Sampling with imblearn
        rus_sampler = RUS(random_state=seed_value, replacement=True)
        train_features_rus, train_target_rus = rus_sampler.fit_resample(
            credit_data.drop(columns=['Class']), credit_data['Class']
        )

        test_features = credit_data.drop(columns=['Class'])
        test_target = credit_data['Class']

        model = build_model(model_type, train_features_rus, train_target_rus)
        predictions = model.predict(test_features)
        sampling_1_scores.append(calculate_performance(metric, test_target, predictions))

        # B) Random Over-Sampling with imblearn
        ros_sampler = ROS(random_state=seed_value)
        train_features_ros, train_target_ros = ros_sampler.fit_resample(
            credit_data.drop(columns=['Class']), credit_data['Class']
        )

        model = build_model(model_type, train_features_ros, train_target_ros)
        predictions = model.predict(test_features)
        sampling_2_scores.append(calculate_performance(metric, test_target, predictions))

        # C) Under-Sampling using Tomek Links
        tl_sampler = TL(sampling_strategy='majority')
        train_features_tl, train_target_tl = tl_sampler.fit_resample(
        credit_data.drop(columns=['Class']), credit_data['Class']
        )

        model = build_model(model_type, train_features_tl, train_target_tl)
        predictions = model.predict(test_features)
        sampling_3_scores.append(calculate_performance(metric, test_target, predictions))

        # D) Synthetic Minority Oversampling Technique (SMOTE)
        smote_sampler = SMOTE()
        train_features_smote, train_target_smote = smote_sampler.fit_resample(
            credit_data.drop(columns=['Class']), credit_data['Class']
        )

        model = build_model(model_type, train_features_smote, train_target_smote)
        predictions = model.predict(test_features)
        sampling_4_scores.append(calculate_performance(metric, test_target, predictions))

        # E) NearMiss
        nm_sampler = NM()
        train_features_nm, train_target_nm = nm_sampler.fit_resample(
            credit_data.drop(columns=['Class']), credit_data['Class']
        )

        model = build_model(model_type, train_features_nm, train_target_nm)
        predictions = model.predict(test_features)
        sampling_5_scores.append(calculate_performance(metric, test_target, predictions))

    result_dataframe = pd.DataFrame({
        'Sampling1': sampling_1_scores,
        'Sampling2': sampling_2_scores,
        'Sampling3': sampling_3_scores,
        'Sampling4': sampling_4_scores,
        'Sampling5': sampling_5_scores
    }, index=model_list)

    return result_dataframe

models_list = ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'KNeighborsClassifier', 'XGBClassifier']
recall_results_dataframe = evaluate_models(models_list, 'recall')
accuracy_results_dataframe = evaluate_models(models_list, 'accuracy')
print(recall_results_dataframe)
print(accuracy_results_dataframe)
recall_results_dataframe.to_csv('recall_results.csv')
accuracy_results_dataframe.to_csv('accuracy_results.csv')