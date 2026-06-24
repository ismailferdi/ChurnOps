from api.core.config import settings
from model.preprocess import load_data, split_data, fit_and_save_preprocessor, transform
from core.features import NUMERIC_FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import numpy as np
import json
import joblib
import mlflow

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment('churn-prediction')

DATASET_PATH = Path(__file__).parent.parent / 'data/raw/telco_churn.csv'

X_train, X_test, y_train, y_test = split_data(load_data(DATASET_PATH))

preprocessor = fit_and_save_preprocessor(X_train, settings.preprocessor_path)

X_train, X_test = transform(preprocessor, X_train), transform(preprocessor, X_test)

reference_distributions = {}
for num_feature in NUMERIC_FEATURES:
    densities, bins = np.histogram(
        X_train,
        bins=20,
        density=True
    )
    reference_distributions[num_feature] = {
        "mean": float(np.mean(X_train)),
        "std": float(np.std(X_train)),
        "bins": bins.tolist(),
        "densities": densities.tolist()
    }

FILE_PATH = Path(settings.reference_distributions_path)

with open(FILE_PATH, 'w',encoding='utf-8') as f:
    json.dump(reference_distributions, f, indent=4)

with mlflow.start_run(run_name='random-forest') as run:
    mlflow.log_param('model_type', 'RandomForestClassifier')
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 'None')
    mlflow.log_param('class_weight', 'balanced')
    mlflow.log_param('random_state', 42)


    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)
    y_prob = rfc.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("roc_auc",roc_auc_score(y_test, y_prob))

    joblib.dump(rfc, settings.model_path)
    mlflow.log_artifact(settings.model_path)
    mlflow.log_artifact(settings.preprocessor_path)
    mlflow.log_artifact(settings.reference_distributions_path)


    print("Run ID: ", run.info.run_id)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision(macro): ",precision_score(y_test, y_pred, average='macro'))
    print("Recall(macro): ",recall_score(y_test, y_pred, average='macro'))
    print("F1(macro): ",f1_score(y_test, y_pred, average='macro'))
    print("ROC AUC: ",roc_auc_score(y_test, y_prob))


with mlflow.start_run(run_name='logistic-regression') as run:
    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.log_param('max_iter', 100)
    mlflow.log_param('class_weight', 'balanced')
    mlflow.log_param('random_state', 42)


    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

    lr.fit(X_train, y_train)

    y_pred = rfc.predict(X_train)
    y_prob = rfc.predict_proba(X_train)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_train, y_pred))
    mlflow.log_metric("precision_macro", precision_score(y_train, y_pred, average='macro'))
    mlflow.log_metric("recall_macro", recall_score(y_train, y_pred, average='macro'))
    mlflow.log_metric("f1_macro", f1_score(y_train, y_pred, average='macro'))
    mlflow.log_metric("roc_auc",roc_auc_score(y_train, y_prob))

    joblib.dump(lr, settings.model_path)
    mlflow.log_artifact(settings.model_path)
    mlflow.log_artifact(settings.preprocessor_path)
    mlflow.log_artifact(settings.reference_distributions_path)


    print("Run ID: ", run.info.run_id)
    print("Accuracy: ", accuracy_score(y_train, y_pred))
    print("Precision(macro): ",precision_score(y_train, y_pred, average='macro'))
    print("Recall(macro): ",recall_score(y_train, y_pred, average='macro'))
    print("F1(macro): ",f1_score(y_train, y_pred, average='macro'))
    print("ROC AUC: ",roc_auc_score(y_train, y_prob))


    










