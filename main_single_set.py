from retrieval import parse_avro_alerts_from_tar
from dataset_builder import build_dataset_from_alerts
from train_model import load_all_batches, train_and_evaluate
from preprocessing import display_sample_images
import sys
import gc

def main():
    original_stdout = sys.stdout
    with open("output_log_single.txt", "w") as f:
        
        sys.stdout = f

        tar_paths = [
            "tarballs\\ztf_public_20250819.tar.gz",
            "tarballs\\ztf_public_20250302.tar.gz",
            "tarballs\\ztf_public_20250920.tar.gz",
            "tarballs\\ztf_public_20251102.tar.gz"
        ]

        batch_idx = 0
        for tar in tar_paths:
            alerts = []
            alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=400000))
            print(f"Loaded {len(alerts)} from {tar}.")
            if batch_idx == 0:
                sys.stdout = original_stdout

                display = int(input("Display first 5 positives and negatives for debugging? (0/1): "))
                if display:
                    display_sample_images(alerts, patch_size=7, output_dir='ztf_pipeline_output_single')
                
                sys.stdout = f
            
            batch_idx = build_dataset_from_alerts(alerts, output_dir='ztf_pipeline_output_single\\batches', desired_patch_size=7, batch_size=200, target_total=600000, batch_idx=batch_idx)
            del alerts
            gc.collect()

        print("Loading all batches...")
        df = load_all_batches("ztf_pipeline_output_single\\batches")
        print(f"Total dataset size: {len(df)} samples.")

        print("Starting training...")
        train_and_evaluate(df, output_dir="ztf_pipeline_output_single")

        sys.stdout = original_stdout
    
    print("Process complete. Check output_log.txt for details.")

if __name__ == "__main__":
    main()