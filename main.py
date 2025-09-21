from retrieval import parse_avro_alerts_from_tar, decode_cutout
from dataset_builder import build_dataset_from_alerts
from train_model import load_all_batches, train_and_evaluate
from preprocessing import preprocess_image, compute_difference, detect_blobs
import matplotlib.pyplot as plt
import numpy as np
import sys

def display_sample_images(alerts, num_samples=5, patch_size=32):
    """Display the first few positive and negative samples from the alerts"""
    pos_count = 0
    neg_count = 0
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15,6))
    fig.suptitle("Sample Images from Alerts", fontsize=16)
    
    for a in alerts:
        if pos_count >= num_samples and neg_count >= num_samples:
            break

        try:
            sci = decode_cutout(a["cutoutScience"])
            ref = decode_cutout(a["cutoutTemplate"]) if a.get("cutoutTemplate") else None
        except Exception as e:
            print(f"Failed to decode cutout: {e}")
            continue
        
        sci_proc = preprocess_image(sci)
        ref_proc = preprocess_image(ref) if ref is not None else None
        diff = compute_difference(sci_proc, ref_proc)
        
        blobs = detect_blobs(diff)
        if blobs and pos_count < num_samples:
            best = min(blobs, key=lambda b: (b.centroid[0] - diff.shape[0]/2)**2 + (b.centroid[1] - diff.shape[1]/2)**2)
            
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
    
    for i in range(pos_count, num_samples):
        axes[0, i].axis('off')
    for i in range(neg_count, num_samples):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    original_stdout = sys.stdout
    with open("output_log.txt", "w") as f:
        
        # sys.stdout = f

        tar_paths = [
            "tarballs\\ztf_public_20250819.tar.gz",
            "tarballs\\ztf_public_20250302.tar.gz"
        ]

        alerts = []
        for tar in tar_paths:
            alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=1000000))
        print(f"Loaded {len(alerts)} alerts.")

        display = int(input("Display first 5 positives and negatives for debugging? (0/1): "))
        if display:
            display_sample_images(alerts, patch_size=7)
        
        sys.stdout = f

        print("Starting dataset building...")
        build_dataset_from_alerts(alerts, output_dir="ztf_pipeline_output", desired_patch_size=7, batch_size=200, target_total=50000)

        print("Loading all batches...")
        df = load_all_batches("ztf_pipeline_output")
        print(f"Total dataset size: {len(df)} samples.")

        print("Starting training...")
        train_and_evaluate(df, output_dir="ztf_pipeline_output")

        sys.stdout = original_stdout
    
    print("Process complete. Check output_log.txt for details.")

if __name__ == "__main__":
    main()