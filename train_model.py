import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

def load_all_batches(input_dir="ztf_pipeline_output"):
    dfs = []
    for f in tqdm(sorted(os.listdir(input_dir)), desc="Loading Batches", file=sys.stderr):
        if f.startswith("batch_") and f.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(input_dir, f)))
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def train_and_evaluate(df, output_dir="ztf_pipeline_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=["label", "alert_id"])
    y = df["label"].astype(int)
    groups = df["alert_id"]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Check for data leakage
    train_alerts = set(groups.iloc[train_idx])
    test_alerts = set(groups.iloc[test_idx])
    common_alerts = train_alerts.intersection(test_alerts)
    if common_alerts:
        print(f"WARNING: {len(common_alerts)} alerts appear in both train and test sets!")
        print("Consider using a different split strategy")
    
    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    dtest = xgb.DMatrix(X_test_s, label=y_test)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    params = {
        "objective" : "binary:logistic",
        "eval_metric" : ["logloss", "aucpr"],
        "tree_method" : "hist",
        "device" : "cuda",
        "verbosity" : 1,
        "scale_pos_weight" : scale_pos_weight,
        "max_depth" : 6,
        "min_child_weight" : 3
    }
    
    start_time = time.time()
    start_lt = time.localtime(start_time)
    start_ft = time.strftime("%Y-%m-%d %H:%M:%S", start_lt)
    print(f"Started model training at: {start_ft}")
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")], early_stopping_rounds=10)
    end_time = time.time()
    end_lt = time.localtime(end_time)
    end_ft = time.strftime("%Y-%m-%d %H:%M:%S", end_lt)
    print(f"Finished model training at: {end_ft}")
    print(f"Time taken: {end_time - start_time} seconds")
    preds = (bst.predict(dtest) >= 0.5).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    print("Precision: ", prec, "Recall: ", rec, "F1: ", f1)
    print(classification_report(y_test, preds))
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bogus', 'Real'])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Asteroid Detection Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))
    print("Saved Confusion Matrix image.")
    plt.close()
    
    bst.save_model(os.path.join(output_dir, "xgb_model.json"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    pd.DataFrame(X_test_s, columns=X.columns).assign(label=y_test.values, pred=preds).to_csv(os.path.join(output_dir, "test_results.csv"))
    
    print("Model, scaler, and test results saved.")

class TransientDetector:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, df_train):
        X = df_train.drop(columns=['label', 'alert_id'])

        self.X_train = self.scaler.fit_transform(X)
        self.y_train = df_train['label'].astype(int)
        
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        params = {
            "objective" : "binary:logistic",
            "eval_metric" : "logloss",
            "tree_method" : "hist",
            "device" : "cuda",
            "verbosity" : 1,
            "scale_pos_weight" : scale_pos_weight,
            "max_depth" : 6,
            "min_child_weight" : 3
        }
        
        self.model = xgb.train(params, dtrain, num_boost_round=200)
        print("Model training complete.")

    def predict(self, df_test):
        X = df_test.drop(columns=['label', 'alert_id'])
        
        self.X_test = self.scaler.transform(X)
        self.y_test = df_test['label'].astype(int)
        
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        pred_probs = self.model.predict(dtest)
        preds = (pred_probs >= 0.5).astype(int)
        print("Predictions complete.")
        
        return preds, pred_probs
    
    def evaluate(self, df_test, preds=None, output_dir='ztf_pipeline_output'):
        os.makedirs(output_dir, exist_ok=True)
        
        y_test = df_test['label'].astype(int)
        
        prec, rec, f1, support = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        print(f"Precision: {prec}\tRecall: {rec}\tF1 Score: {f1}\tSupport: {support}")
        print(classification_report(y_test, preds))
        
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bogus', 'Real'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Asteroid Detection Confusion Matrix')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))
        print("Saved Confusion Matrix image.")
        plt.close()
    
    def save_model(self, output_dir='ztf_pipeline_output'):
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_model(os.path.join(output_dir, 'xgb_model.json'))
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        print("Model and scaler saved.")
    
    def save_predictions(self, df_test, pred_probs, preds, output_dir='ztf_pipeline_output'):
        os.makedirs(output_dir, exist_ok=True)
        
        X = df_test.drop(columns=['label', 'alert_id'])
        results_df = pd.DataFrame(self.X_test, columns=X.columns).assign(
            label=self.y_test.values,
            pred_prob=pred_probs,
            pred=preds
        )
        results_df.to_csv(os.path.join(output_dir, 'test_results.csv'))
        print("Test results saved.")
    
    def run_all(self, df_train, df_test, output_dir='ztf_pipeline_output'):
        self.fit(df_train)
        preds, pred_probs = self.predict(df_test)
        self.evaluate(df_test, preds, output_dir)
        self.save_model(output_dir)
        self.save_predictions(df_test, pred_probs, preds, output_dir)