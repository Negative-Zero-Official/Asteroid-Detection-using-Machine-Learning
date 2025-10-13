import os
from pyspark.sql import SparkSession, functions as F, types as T, udf
import pandas as pd
from preprocessing import (
    preprocess_image_udf, 
    compute_difference_udf, 
    extract_features_udf, 
    extract_negative_features_udf
)

def build_dataset_from_alerts_spark(
    alerts_df,
    output_dir="ztf_pipeline_output_spark",
    n_random_neg_per_alert=1,
    desired_patch_size=32,
    target_total=5000000,
    neg_threshold_sigma=3.0,
    min_distance_from_center=12
):
    """
    Build dataset using PySpark for distributed processing
    """
    print("Starting Spark dataset building...")
    
    # Apply preprocessing pipeline
    processed_df = alerts_df \
        .withColumn("sci_proc", preprocess_image_udf(F.col("science_array"))) \
        .withColumn("ref_proc", preprocess_image_udf(F.col("template_array"))) \
        .withColumn("diff", compute_difference_udf(F.col("sci_proc"), F.col("ref_proc")))
    
    print("Preprocessing completed")
    
    # Extract positive features (from blobs)
    positive_df = processed_df \
        .withColumn("positive_features", extract_features_udf(F.col("diff"), F.col("sci_proc"))) \
        .filter(F.col("positive_features").isNotNull()) \
        .select(
            F.col("ra"),
            F.col("dec"), 
            F.col("jd"),
            F.col("positive_features.*")
        )
    
    positive_count = positive_df.count()
    print(f"Extracted {positive_count} positive samples")
    
    # Extract negative features
    negative_df = processed_df \
        .withColumn(
            "negative_features", 
            extract_negative_features_udf(
                F.col("diff"),
                F.lit(desired_patch_size),
                F.lit(neg_threshold_sigma),
                F.lit(min_distance_from_center)
            )
        ) \
        .filter(F.col("negative_features").isNotNull()) \
        .select(
            F.col("ra"),
            F.col("dec"),
            F.col("jd"), 
            F.col("negative_features.*")
        )
    
    # Sample negatives if we have too many
    if positive_count > 0:
        # Calculate how many negatives we need to reach target_total with balanced classes
        negatives_needed = min(target_total - positive_count, positive_count * n_random_neg_per_alert)
        
        if negatives_needed > 0:
            negative_count = negative_df.count()
            fraction = min(1.0, negatives_needed / negative_count) if negative_count > 0 else 0
            
            if fraction > 0:
                negative_df = negative_df.sample(withReplacement=False, fraction=fraction)
            else:
                negative_df = negative_df.limit(negatives_needed)
    
    negative_count = negative_df.count()
    print(f"Extracted {negative_count} negative samples")
    
    # Combine positive and negative samples
    final_dataset = positive_df.union(negative_df)
    
    total_count = final_dataset.count()
    print(f"Final dataset size: {total_count} samples")
    
    # Write to parquet with multiple partitions for better parallelism
    final_dataset.repartition(max(1, total_count // 10000)) \
        .write \
        .mode("overwrite") \
        .option("maxRecordsPerFile", 10000) \
        .parquet(output_dir)
    
    print(f"Dataset complete: {total_count} samples saved in {output_dir}")
    
    return final_dataset

def build_dataset_from_alerts(
    alerts,
    output_dir="ztf_pipeline_output",
    n_random_neg_per_alert=1,
    desired_patch_size=32,
    batch_size=200,
    target_total=5000000,
    max_attempts=50,
    neg_threshold_sigma=3.0,
    min_distance_from_center=12
):
    """
    Legacy function for backward compatibility
    """
    print("WARNING: Using legacy single-node dataset builder. Consider using Spark version.")
    
    # Convert alerts to Spark DataFrame and use Spark version
    spark = SparkSession.builder.appName("LegacyDatasetBuilder").getOrCreate()
    
    try:
        # Convert alerts to Spark DataFrame
        alerts_df = spark.createDataFrame(alerts)
        
        # Use Spark version
        return build_dataset_from_alerts_spark(
            alerts_df,
            output_dir=output_dir,
            n_random_neg_per_alert=n_random_neg_per_alert,
            desired_patch_size=desired_patch_size,
            target_total=target_total,
            neg_threshold_sigma=neg_threshold_sigma,
            min_distance_from_center=min_distance_from_center
        )
    finally:
        spark.stop()