import joblib
from pathlib import Path
from model.preprocess import split_data, load_data, transform
from api.core.config import settings
from sklearn.metrics import classification_report, roc_auc_score

model = joblib.load(settings.model_path)
preprocessor = joblib.load(settings.preprocessor_path)
DATASET_PATH = Path(__file__).parent.parent / "data/raw/telco_churn.csv"
X_train, X_test, y_train, y_test = split_data(load_data(DATASET_PATH))
X_train, X_test = transform(preprocessor, X_train), transform(preprocessor, X_test)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_prob[:, 1]))
