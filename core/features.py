FEATURE_ORDER = ["gender",
                 "SeniorCitizen",
                 "Partner",
                 "Dependents",
                 "tenure",
                 "PhoneService",
                 "MultipleLines",
                 "InternetService",
                 "OnlineSecurity",
                 "OnlineBackup",
                 "DeviceProtection",
                 "TechSupport",
                 "StreamingTV",
                 "StreamingMovies",
                 "Contract",
                 "PaperlessBilling",
                 "PaymentMethod",
                 "MonthlyCharges",
                 "TotalCharges"]

NUMERIC_FEATURES = ["tenure",
                    "MonthlyCharges",
                    "TotalCharges"]

BINARY_CATEGORICAL_FEATURES = ["gender",
                               "SeniorCitizen",
                               "Partner",
                               "Dependents",
                               "PhoneService",
                               "PaperlessBilling"]

MULTI_CATEGORICAL_FEATURES = ["MultipleLines",
                              "InternetService",
                              "OnlineSecurity",
                              "OnlineBackup",
                              "DeviceProtection",
                              "TechSupport",
                              "StreamingTV",
                              "StreamingMovies",
                              "Contract",
                              "PaymentMethod"]

PASSTHROUGH_FEATURES = ["SeniorCitizen",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges"]

TARGET = "Churn"