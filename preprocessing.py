import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import hog
from skimage.measure import label, regionprops
from skimage.transform import resize

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
