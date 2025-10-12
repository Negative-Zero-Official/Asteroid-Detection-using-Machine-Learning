import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Set display options to show all rows/columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def load_all_batches(output_dir="ztf_pipeline_output_spark"):
    """Load all parquet files from Spark output directory"""
    spark = SparkSession.builder.appName("DataLoader").getOrCreate()
    
    try:
        # Read all parquet files from the directory
        df_spark = spark.read.parquet(output_dir)
        
        # Convert to pandas (if data fits in memory)
        print("Converting Spark DataFrame to pandas...")
        df_pandas = df_spark.toPandas()
        
        print(f"Loaded {len(df_pandas)} samples")
        return df_pandas
    except Exception as e:
        print(f"Error loading data with Spark: {e}")
        # Fallback to original method for non-Spark outputs
        return load_all_batches_legacy(output_dir)
    finally:
        spark.stop()

def load_all_batches_legacy(output_dir="ztf_pipeline_output"):
    """Legacy method for loading parquet files without Spark"""
    dfs = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("batch_") and f.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(output_dir, f)))
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def train_and_evaluate(df, output_dir="ztf_pipeline_output"):
    """Train and evaluate model (same as original)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both Spark-generated and legacy data
    required_columns = ["ra", "dec", "jd", "label"]
    feature_columns = [col for col in df.columns if col not in required_columns and col != "alert_id"]
    
    # If alert_id doesn't exist, create one from index
    if "alert_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "alert_id"})
    
    X = df[feature_columns]
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
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 1
    }
    
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")], early_stopping_rounds=10)
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
    
    # Save test results
    test_results = pd.DataFrame(X_test_s, columns=X.columns)
    test_results = test_results.assign(label=y_test.values, pred=preds)
    test_results.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
    
    print("Model, scaler, and test results saved.")

def train_with_spark(df_spark, output_dir="ztf_pipeline_output"):
    """
    Alternative training function that works directly with Spark DataFrames
    for very large datasets (uses Spark ML)
    """
    from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql.functions import col
    
    print("Training with Spark ML...")
    
    # Prepare features
    feature_columns = [c for c in df_spark.columns if c not in ["ra", "dec", "jd", "label", "alert_id"]]
    
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_assembled = assembler.transform(df_spark)
    
    # Split data
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100)
    
    # Cross-validation (optional)
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 10]) \
        .addGrid(gbt.maxBins, [32, 64]) \
        .build()
    
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    
    crossval = CrossValidator(estimator=gbt,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)
    
    cvModel = crossval.fit(train_df)
    
    # Make predictions
    predictions = cvModel.transform(test_df)
    
    # Evaluate
    accuracy = predictions.filter(col("label") == col("prediction")).count() / float(test_df.count())
    auc = evaluator.evaluate(predictions)
    
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    
    # Save model
    cvModel.bestModel.write().overwrite().save(os.path.join(output_dir, "spark_gbt_model"))
    
    return cvModel.bestModel