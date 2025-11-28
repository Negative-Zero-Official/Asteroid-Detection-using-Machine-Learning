import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from skimage.feature import hog
from skimage.transform import resize
from astropy.stats import sigma_clipped_stats
from retrieval import decode_cutout

def normalize_image(arr, clip_sigma=5.0, to_range=True):
    finite = arr[np.isfinite(arr)]
    median_f, max_f, min_f = 0.0, 0.0, 0.0
    if finite.size:
        median_f = np.median(finite)
        max_f = np.max(finite)
        min_f = np.min(finite)
    
    arr = np.nan_to_num(arr, nan=median_f, posinf=max_f, neginf=min_f).astype(np.float32)
    
    _, median, std = sigma_clipped_stats(arr, sigma=5.0, maxiters=5)
    bg = median
    scale = std if (std is not None and std > 1e-6) else 1.0
    
    norm = (arr - bg) / scale
    norm = np.clip(norm, -clip_sigma, clip_sigma)
    
    if to_range:
        norm = (norm + clip_sigma) / (2 * clip_sigma)
    
    return norm

def extract_features_from_image(image, img_type='sci'):
    if image is None:
        return None
    
    arr = np.asarray(image, dtype=np.float32)
    
    norm = normalize_image(arr, clip_sigma=5.0, to_range=True)
    
    feats = {
        img_type + '_mean' : float(np.mean(norm)),
        img_type + '_std' : float(np.std(norm)),
        img_type + '_max' : float(np.max(norm))
    }
    
    resized = resize(norm, (63, 63), anti_aliasing=True)
    hog_vec = hog(resized, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
    for i, feat in enumerate(hog_vec):
        feats[img_type + f"_hog_{i}"] = feat
    
    return feats

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
        
        # if 0.3 <= reality_score <= 0.7:
        #     continue
        
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
        
        combined_feat = {
            'alert_id' : i,
            'magpsf' : a['magpsf'],
            'sigmapsf' : a['sigmapsf'],
            'fwhm' : a['fwhm'],
            'ndethist' : a['ndethist'],
            'sgscore' : a['sgscore'],
            'ssdistnr' : a['ssdistnr'],
            'label' : 1 if reality_score >= 0.85 else 0
        }
        
        for img_type, img in [('sci', sci), ('ref', ref), ('diff', diff)]:
            if img is not None:
                feat = extract_features_from_image(img, img_type)
                if feat is not None:
                    combined_feat.update(feat)
        
        if 'sci_mean' in combined_feat and 'ref_mean' in combined_feat and 'diff_mean' in combined_feat:
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