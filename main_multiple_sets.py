from retrieval import parse_avro_alerts_from_tar
from dataset_builder import build_dataset_from_alerts
from train_model import load_all_batches, TransientDetector
from preprocessing import display_sample_images
import sys
import gc

def main():
    original_stdout = sys.stdout
    together = int(input("Test all testing data together? (0/1): "))
    with open('output_log_multiple.txt', 'w') as f:
        sys.stdout = f
        
        training_tar_paths = [
            "tarballs\\ztf_public_20250920.tar.gz"
        ]
        
        batch_idx = 0
        for tar in training_tar_paths:
            alerts = []
            alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=400000))
            print(f"Loaded {len(alerts)} from {tar} (training).")
            if batch_idx == 0:
                display_sample_images(alerts, num_samples=5, patch_size=7, output_dir='ztf_pipeline_output_multiple')
            batch_idx = build_dataset_from_alerts(alerts, output_dir='ztf_pipeline_output_multiple\\batches\\train', desired_patch_size=7, batch_size=200, target_total=600000, batch_idx=batch_idx)
            del alerts
            gc.collect()
        
        print("Loading training batches...")
        df_train = load_all_batches('ztf_pipeline_output_multiple\\batches\\train')
        print(f"Total training dataset size: {len(df_train)} samples.")
        
        print("Spawning Transient Detector model and fitting data...")
        model = TransientDetector()
        model.fit(df_train)
        
        testing_tar_paths = [
            "tarballs\\ztf_public_20250302.tar.gz",
            "tarballs\\ztf_public_20250819.tar.gz",
            "tarballs\\ztf_public_20251102.tar.gz"
        ]
        
        if not together:
            for i, tar in enumerate(testing_tar_paths):
                alerts = []
                alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=400000))
                print(f"Loaded {len(alerts)} from {tar} (testing).")
                batch_idx = build_dataset_from_alerts(alerts, output_dir=f'ztf_pipeline_output_multiple\\batches\\test_{i}', desired_patch_size=7, batch_size=200, target_total=600000, batch_idx=0)
                del alerts
                gc.collect()
            
            print("Testing each training dataset separately...")
            
            for i, tar in enumerate(testing_tar_paths):
                test_batch_dir = f'ztf_pipeline_output_multiple\\batches\\test_{i}'
                
                print(f"Loading dataset for TAR {tar}...")
                df_test = load_all_batches(test_batch_dir)
                print(f"Test dataset size for {tar}: {len(df_test)} samples.")
                
                print(f"Running predictions for TAR {tar}...")
                preds, pred_probs = model.predict(df_test)
                
                test_output_dir = f'ztf_pipeline_output_multiple\\eval_test_{i}'
                print(f"Evaluating model for TAR {tar}...")
                model.evaluate(df_test, preds, output_dir=test_output_dir)
                
                print(f"Saving predictions for TAR {tar}...")
                model.save_predictions(df_test, pred_probs, preds, test_output_dir)
            
        else:
            batch_idx = 0
            for tar in testing_tar_paths:
                alerts = []
                alerts.extend(parse_avro_alerts_from_tar(tar_path=tar, max_alerts=400000))
                print(f"Loaded {len(alerts)} from {tar} (testing).")
                batch_idx = build_dataset_from_alerts(alerts, output_dir='ztf_pipeline_output_multiple\\batches\\test', desired_patch_size=7, batch_size=200, target_total=600000, batch_idx=batch_idx)
                del alerts
                gc.collect()
            
            print("Loading testing batches...")
            df_test = load_all_batches('ztf_pipeline_output_multiple\\batches\\test')
            print(f"Total testing dataset size: {len(df_test)} samples.")
            
            print("Spawning Transient Detector model and fitting data...")
            model = TransientDetector()
            model.fit(df_train)
            
            print("Running predictions on test data...")
            preds, pred_probs = model.predict(df_test)
            
            print("Evaluating performance...")
            model.evaluate(df_test, preds, output_dir='ztf_pipeline_output_multiple')
            
            print("Saving predictions...")
            model.save_predictions(df_test, pred_probs, preds, output_dir='ztf_pipeline_output_multiple')
        
        print("Saving model...")
        model.save_model('ztf_pipeline_output_multiple')
                
        sys.stdout = original_stdout
    
    print("Process complete. Check output_log.txt for details.")

if __name__ == "__main__":
    main()