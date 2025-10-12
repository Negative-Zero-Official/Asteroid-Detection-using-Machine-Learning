from retrieval import process_alerts_with_spark
from dataset_builder import build_dataset_from_alerts_spark
from train_model import load_all_batches, train_and_evaluate
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def display_sample_images(spark, alerts_df, num_samples=5, patch_size=32):
    """Display the first few positive and negative samples from the alerts"""
    # Take a sample of the data for visualization
    sample_df = alerts_df.limit(num_samples * 10)  # Get more to ensure we find samples
    
    # Collect science arrays for display
    sample_data = sample_df.select("science_array", "template_array").collect()
    
    pos_count = 0
    neg_count = 0
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15,6))
    fig.suptitle("Sample Images from Alerts", fontsize=16)
    
    for row in sample_data:
        if pos_count >= num_samples and neg_count >= num_samples:
            break

        try:
            sci_array = row["science_array"]
            ref_array = row["template_array"]
            
            if sci_array is None:
                continue
                
            sci = np.array(sci_array, dtype=np.float32)
            ref = np.array(ref_array, dtype=np.float32) if ref_array else None
            
            # Simple preprocessing for display
            sci_proc = (sci - np.median(sci)) / (np.std(sci) + 1e-8)
            ref_proc = (ref - np.median(ref)) / (np.std(ref) + 1e-8) if ref is not None else None
            
            diff = sci_proc - ref_proc if ref_proc is not None else sci_proc
            
            # Simple blob detection for display
            med = np.median(diff)
            std = np.std(diff)
            threshold = med + 5.0 * std
            mask = diff > threshold
            
            labeled_mask = np.zeros_like(mask, dtype=int)
            labeled_mask[mask] = 1
            
            # Find connected components
            from skimage.measure import label, regionprops
            lbl = label(labeled_mask)
            props = regionprops(lbl, intensity_image=diff)
            blobs = [p for p in props if p.area >= 3]
            
            if blobs and pos_count < num_samples:
                h, w = diff.shape
                cx, cy = w / 2.0, h / 2.0
                best = min(blobs, key=lambda b: (b.centroid[0] - cy)**2 + (b.centroid[1] - cx)**2)
                
                y0, x0, y1, x1 = best.bbox
                patch = diff[y0:y1, x0:x1]
                
                axes[0, pos_count].imshow(patch, cmap='gray')
                axes[0, pos_count].set_title(f"Positive #{pos_count+1}")
                axes[0, pos_count].axis('off')
                pos_count += 1
            
            if neg_count < num_samples:
                for _ in range(10):
                    x0 = np.random.randint(0, diff.shape[1] - patch_size)
                    y0 = np.random.randint(0, diff.shape[0] - patch_size)
                    dist = np.sqrt((x0 + patch_size/2 - diff.shape[1]/2)**2 + (y0 + patch_size/2 - diff.shape[0]/2)**2)
                    
                    if dist > 12:
                        patch = diff[y0:y0+patch_size, x0:x0+patch_size]
                        axes[1, neg_count].imshow(patch, cmap='gray')
                        axes[1, neg_count].set_title(f"Negative #{neg_count+1}")
                        axes[1, neg_count].axis('off')
                        neg_count += 1
                        break
                    
        except Exception as e:
            print(f"Failed to process sample: {e}")
            continue
    
    for i in range(pos_count, num_samples):
        axes[0, i].axis('off')
    for i in range(neg_count, num_samples):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    original_stdout = sys.stdout
    
    # Initialize Spark
    from retrieval import get_spark_session
    spark = get_spark_session()
    
    with open("output_log.txt", "w") as f:
        tar_paths = [
            "tarballs/ztf_public_20250819.tar.gz",
            "tarballs/ztf_public_20250302.tar.gz",
            "tarballs/ztf_public_20250920.tar.gz"
        ]

        print("Loading alerts with Spark...")
        alerts_df = process_alerts_with_spark(tar_paths, max_alerts_per_tar=100000)
        print(f"Loaded {alerts_df.count()} alerts.")

        display = int(input("Display first 5 positives and negatives for debugging? (0/1): "))
        if display:
            display_sample_images(spark, alerts_df, patch_size=7)
        
        sys.stdout = f

        print("Starting dataset building with Spark...")
        build_dataset_from_alerts_spark(
            alerts_df, 
            output_dir="ztf_pipeline_output_spark", 
            desired_patch_size=7, 
            target_total=500000
        )

        print("Loading all batches...")
        df = load_all_batches("ztf_pipeline_output_spark")
        print(f"Total dataset size: {len(df)} samples.")

        print("Starting training...")
        train_and_evaluate(df, output_dir="ztf_pipeline_output")

        sys.stdout = original_stdout
    
    # Stop Spark session
    spark.stop()
    
    print("Process complete. Check output_log.txt for details.")

if __name__ == "__main__":
    main()