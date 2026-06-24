from api.models.requests import CustomerFeatures
from api.core.errors import PreprocessingError, PredictionError


def predict(customer: CustomerFeatures, model, preprocessor) -> dict:



    df = customer.to_dataframe()
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise PreprocessingError(
            internal_message=str(e),
            client_message="Unable to process the provided customer data."
            )
    
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
    except Exception as e:
        raise PredictionError(
            internal_message=f"Error: {e}.",
            client_message="Unable to generate a prediction."
            )
    
    churn_probability = round(float(y_prob[0][1]), 4)
    churn_prediction = int(y_pred[0])
    prediction_label = "Will Churn" if churn_prediction == 1 else "Will Not Churn"
    return {"churn_prediction": churn_prediction,
            "churn_probability": churn_probability,
            "prediction_label": prediction_label
            }