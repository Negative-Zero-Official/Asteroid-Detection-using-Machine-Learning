import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import hog
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
from retrieval import decode_cutout
import os

def preprocess_image(img, filter_size=3):
    img_f = median_filter(img, size=filter_size)
    norm = (img_f - np.median(img_f)) / (np.std(img_f) + 1e-8)
    return norm

def compute_difference(science_img, reference_img):
    return science_img - reference_img if reference_img is not None else science_img

def detect_blobs(diff_img, threshold_sigma=5.0):
    med = np.median(diff_img)
    std = np.std(diff_img)
    threshold = med + threshold_sigma * std
    mask = diff_img > threshold
    lbl = label(mask)
    props = regionprops(lbl, intensity_image=diff_img)
    return [p for p in props if p.area >= 3]

def extract_features_from_blob(blob, diff_img, sci_img):
    y0, x0, y1, x1 = blob.bbox
    patch_diff = diff_img[y0:y1, x0:x1]
    
    mean_patch = float(np.mean(patch_diff))
    std_patch = float(np.std(patch_diff))
    
    feats = {
        "mean_diff" : mean_patch,
        "std_diff" : std_patch,
    }
    
    small = resize(patch_diff, (32, 32), anti_aliasing=True)
    hog_vec = hog(small, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, feature_vector=True)
    for i in range(12):
        feats[f"hog_{i}"] = float(hog_vec[i] if i < len(hog_vec) else 0.0)
    # for i, feat in enumerate(hog_vec):
    #     feats[f"hog_{i}"] = feat
    
    return feats

def display_sample_images(alerts, num_samples=5, patch_size=32, output_dir='ztf_pipeline_output'):
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.show()