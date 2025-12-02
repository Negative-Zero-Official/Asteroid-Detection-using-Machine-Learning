import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from retrieval import decode_cutout
from feature_extractor import extract_alert_features

def build_dataset_from_alerts(
    alerts,
    output_dir='ztf_pipeline_output',
    batch_size=200,
    target_total=5000000,
    batch_idx=0
):
    os.makedirs(output_dir, exist_ok=True)
    features = []
    count = 0
    
    print(f"Starting dataset building... ({batch_idx:03d})")
    for i, a in enumerate(tqdm(alerts, desc='Dataset Building', file=sys.stderr)):
        if count >= target_total:
            break

        reality_score = a['drb']
        
        try:
            sci = decode_cutout(a['cutoutScience'])
        except Exception:
            continue
        ref = None
        if a.get('cutoutTemplate'):
            try:
                ref = decode_cutout(a['cutoutTemplate'])
            except Exception:
                ref = None
        try:
            diff = decode_cutout(a['cutoutDifference'])
        except Exception:
            continue
        
        fwhm_pixels = a['fwhm'] / 1.01
        
        alert_feats = extract_alert_features(
            sci=sci, ref=ref, diff=diff,
            fwhm_alert=fwhm_pixels
        )
        
        combined_feat = {
            'alert_id' : i,
            'magpsf' : a['magpsf'],
            'sigmapsf' : a['sigmapsf'],
            'fwhm' : a['fwhm'],
            'ndethist' : a['ndethist'],
            'sgscore1' : a['sgscore1'],
            'sgscore2' : a['sgscore2'],
            'sgscore3' : a['sgscore3'],
            'ssdistnr' : a['ssdistnr'],
            'label' : 1 if reality_score >= 0.85 else 0
        }
        
        combined_feat.update(alert_feats)
        features.append(combined_feat)
        count += 1
        
        if len(features) > batch_size:
            df_feats = pd.DataFrame(features)
            outpath = os.path.join(output_dir, f'batch_{batch_idx:03d}.parquet')
            df_feats.to_parquet(outpath, index=False)
            features = []
            batch_idx += 1
    
    if features:
        df_feats = pd.DataFrame(features)
        outpath = os.path.join(output_dir, f'batch_{batch_idx:03d}.parquet')
        df_feats.to_parquet(outpath, index=False)
        print(f"Completed and saved FINAL batch {batch_idx} to {outpath}")
    
    print(f"Dataset complete: {count} samples saved in {output_dir}")
    return batch_idx + 1