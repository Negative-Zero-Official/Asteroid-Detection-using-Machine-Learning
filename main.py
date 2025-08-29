from retrieval import parse_avro_alerts_from_tar
from dataset_builder import build_dataset_from_alerts
from train_model import load_all_batches, train_and_evaluate
import sys

def main():
    original_stdout = sys.stdout
    with open("output_log.txt", "w") as f:
        sys.stdout = f

        tar_paths = [
            "tarballs\\ztf_public_20250819.tar.gz",
            "tarballs\\ztf_public_20250302.tar.gz"
        ]

        alerts = []
        for tar in tar_paths:
            alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=1000))
        print(f"Loaded {len(alerts)} alerts.")

        print("Starting dataset building...")
        build_dataset_from_alerts(alerts, output_dir="ztf_pipeline_output", batch_size=200, target_total=50000)

        print("Loading all batches...")
        df = load_all_batches("ztf_pipeline_output")
        print(f"Total dataset size: {len(df)} samples.")

        print("Starting training...")
        train_and_evaluate(df, output_dir="ztf_pipeline_output")

        sys.stdout = original_stdout
    
    print("Process complete. Check output_log.txt for details.")

if __name__ == "__main__":
    main()