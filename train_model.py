import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Set display options to show all rows/columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full column content

def load_all_batches(output_dir="ztf_pipeline_output"):
    dfs = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("batch_") and f.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(output_dir, f)))
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def train_and_evaluate(df, output_dir="ztf_pipeline_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=["ra", "dec", "jd", "label", "alert_id"])
    y = df["label"].astype(int)
    groups = df["alert_id"]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Debugging Print Statements
    # print(X_train)
    # print("=" * 50)
    # print(X_test)
    # print("=" * 50)
    # print(y_train)
    # print("=" * 50)
    # print(y_test)
    
    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    dtest = xgb.DMatrix(X_test_s, label=y_test)
    
    params = {
        "objective" : "binary:logistic",
        "eval_metric" : "logloss",
        "tree_method" : "hist",
        "device" : "cuda",
        "verbosity" : 1
    }
    
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")])
    preds = (bst.predict(dtest) >= 0.5).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    print("Precision: ", prec, "Recall: ", rec, "F1: ", f1)
    print(classification_report(y_test, preds))
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Asteroid Detection Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))
    print("Saved Confusion Matrix image.")
    plt.close()
    
    bst.save_model(os.path.join(output_dir, "xgb_model.json"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    pd.DataFrame(X_test_s, columns=X.columns).assign(label=y_test.values, pred=preds).to_csv(os.path.join(output_dir, "test_results.csv"))
    
    print("Model, scaler, and test results saved.")