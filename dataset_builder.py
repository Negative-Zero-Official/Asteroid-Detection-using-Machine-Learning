import os
import numpy as np
import pandas as pd
from math import hypot
from retrieval import decode_cutout
from preprocessing import preprocess_image, compute_difference, detect_blobs, extract_features_from_blob

def build_dataset_from_alerts(
    alerts,
    output_dir="ztf_pipeline_output",
    n_random_neg_per_alert=1,
    desired_patch_size=32,
    batch_size=200,
    target_total=50000,
    max_attempts=50,
    neg_threshold_sigma=3.0,
    min_distance_from_center=12
):
    os.makedirs(output_dir, exist_ok=True)
    features = []
    meta = []
    count = 0
    batch_idx = 0
    
    print("Starting dataset building...")
    for i, a in enumerate(alerts):
        if count >= target_total:
            break
        print(f"Processing alert {i+1}/{len(alerts)}    (collected {count}/{target_total})")
        
        try:
            sci = decode_cutout(a["cutoutScience"])
        except Exception as e:
            print(f"WARNING: failed to decode science cutout for alert {i+1}: {e}")
            continue
        ref = None
        if a.get("cutoutTemplate"):
            try:
                ref = decode_cutout(a["cutoutTemplate"])
            except Exception:
                ref = None
        
        print("Preprocessing images...")
        sci_proc = preprocess_image(sci)
        ref_proc = preprocess_image(ref) if ref is not None else None
        print("Computing difference...")
        diff = compute_difference(sci_proc, ref_proc)
        
        h, w = diff.shape
        cx, cy = w / 2.0, h / 2.0
        
        print("Detecting blobs...")
        blobs = detect_blobs(diff)
        if blobs:
            best = min(blobs, key=lambda b: (b.centroid[0] - cy)**2 + (b.centroid[1] - cx)**2)
            feat = extract_features_from_blob(best, diff, sci_proc)
            feat["label"] = 1
            features.append(feat)
            meta.append({
                "ra" : a.get("ra"),
                "dec" : a.get("dec"),
                "jd" : a.get("jd"),
                "alert_id" : i
            })
            count += 1
        
        patch_size = min(desired_patch_size, h, w)
        allow_full_stamp = (h == patch_size and w == patch_size)
        
        global_med = np.median(diff)
        global_std = np.std(diff)
        accept_threshold = global_med + neg_threshold_sigma * global_std

        negs_added = 0
        for nn in range(n_random_neg_per_alert):
            best_candidate = None
            best_candidate_score = np.inf
            accepted = False
            attempts = 0
            
            while attempts < max_attempts and not accepted:
                attempts += 1
                if allow_full_stamp:
                    x0, y0 = 0, 0
                else:
                    max_x = max(0, w - patch_size)
                    max_y = max(0, h - patch_size)
                    if max_x == 0 and max_y == 0:
                        x0, y0 = 0, 0
                    else:
                        x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0
                        y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
                
                sub = diff[y0:y0 + patch_size, x0:x0 + patch_size]
                if sub.size == 0:
                    continue
                
                px_cx = x0 + patch_size / 2.0
                px_cy = y0 + patch_size / 2.0
                dist_to_center = hypot(px_cx - cx, px_cy - cy)
                
                if dist_to_center < min_distance_from_center:
                    score = np.max(sub)
                    if score < best_candidate_score:
                        best_candidate_score = score
                        best_candidate = (x0, y0, sub.copy())
                    continue

                local_max = np.max(sub)
                if local_max < accept_threshold:
                    sub_sci = sci_proc[y0:y0 + patch_size, x0:x0 + patch_size]
                    feat = {
                        "area" : np.random.uniform(low=0.0, high=5.0),
                        "eccentricity" : np.random.uniform(low=0.0, high=1.0),
                        "solidity" : np.random.uniform(0.6, 1.0),
                        "orientation" : np.random.uniform(-1.57, 1.57),
                        "total_flux" : float(np.sum(sub_sci)),
                        "peak_flux" : float(np.max(sub_sci)),
                        "mean_diff" : float(np.median(sub)),
                        "std_diff" : float(np.std(sub)),
                        "snr" : np.random.uniform(-0.5, 3.0)
                    }
                    for k in range(12):
                        feat[f"hog_{k}"] = np.random.uniform(0.0, 0.2)
                    feat["label"] = 0
                    features.append(feat)
                    meta.append({
                        "ra" : None,
                        "dec" : None,
                        "jd" : a.get("jd"),
                        "alert_id" : i
                    })
                    negs_added += 1
                    count += 1
                    accepted = True
                    break

                else:
                    if local_max < best_candidate_score:
                        best_candidate_score = local_max
                        best_candidate = (x0, y0, sub.copy())
            
            if not accepted:
                if best_candidate is not None:
                    x0, y0, sub = best_candidate
                    sub_sci = sci_proc[y0:y0 + patch_size, x0:x0 + patch_size]
                    feat = {
                        "area" : np.random.uniform(low=0.0, high=5.0),
                        "eccentricity" : np.random.uniform(low=0.0, high=1.0),
                        "solidity" : np.random.uniform(0.6, 1.0),
                        "orientation" : np.random.uniform(-1.57, 1.57),
                        "total_flux" : float(np.sum(sub_sci)),
                        "peak_flux" : float(np.max(sub_sci)),
                        "mean_diff" : float(np.median(sub)),
                        "std_diff" : float(np.std(sub)),
                        "snr" : np.random.uniform(-0.5, 3.0)
                    }
                    for k in range(12):
                        feat[f"hog_{k}"] = np.random.uniform(0.0, 0.2)
                    feat["label"] = 0
                    features.append(feat)
                    meta.append({
                        "ra" : None,
                        "dec" : None,
                        "jd" : a.get("jd"),
                        "alert_id" : i
                    })
                    negs_added += 1
                    count += 1
                    print(f"WARNING: Used fallback negative patch after {attempts} attempts (alert {i+1})")
                else:
                    print(f"WARNING: Could not generate any negative patches (alert {i+1})")
        
        if len(features) >= batch_size:
            df_feats = pd.DataFrame(features)
            df_meta = pd.DataFrame(meta)
            df_combined = pd.concat([df_meta.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)
            rows_saved = len(df_combined)
            outpath = os.path.join(output_dir, f"batch_{batch_idx:03d}.parquet")
            df_combined.to_parquet(outpath, index=False)
            print(f"Completed and saved batch {batch_idx} with {rows_saved} rows to {outpath}")
            features = []
            meta = []
            batch_idx += 1
    
    if features:
        df_feats = pd.DataFrame(features)
        df_meta = pd.DataFrame(meta)
        df_combined = pd.concat([df_meta.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)
        rows_saved = len(df_combined)
        outpath = os.path.join(output_dir, f"batch_{batch_idx:03d}.parquet")
        df_combined.to_parquet(outpath, index=False)
        print(f"Completed and saved FINAL batch {batch_idx} with {rows_saved} rows to {outpath}")
    
    print(f"Dataset complete: {count} samples saved in {output_dir}")