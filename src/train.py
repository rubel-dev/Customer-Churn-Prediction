import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import get_data, preprocess_data

def train_model():
    df = get_data()
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(" Model Evaluation Report:\n")
    print(classification_report(y_test, y_pred))
    print(" ROC AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(clf, "models/churn_model.pkl")
    print(" Model saved to models/churn_model.pkl")

if __name__ == "__main__":
    train_model()
