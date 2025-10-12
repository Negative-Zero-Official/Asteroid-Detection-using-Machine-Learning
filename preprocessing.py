from pyspark.sql import functions as F, types as T
from pyspark.sql.udf import udf
import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import hog
from skimage.measure import label, regionprops
from skimage.transform import resize

# UDF for preprocessing images
@udf(T.ArrayType(T.ArrayType(T.FloatType())))
def preprocess_image_udf(img_array, filter_size=3):
    """Preprocess image with median filtering and normalization"""
    if img_array is None:
        return None
    try:
        img = np.array(img_array, dtype=np.float32)
        img_f = median_filter(img, size=filter_size)
        norm = (img_f - np.median(img_f)) / (np.std(img_f) + 1e-8)
        return norm.tolist()
    except Exception as e:
        return None

# UDF for computing difference
@udf(T.ArrayType(T.ArrayType(T.FloatType())))
def compute_difference_udf(science_img, reference_img):
    """Compute difference between science and reference images"""
    if science_img is None:
        return None
    
    sci_array = np.array(science_img, dtype=np.float32)
    if reference_img is None:
        return sci_array.tolist()
    
    ref_array = np.array(reference_img, dtype=np.float32)
    diff = sci_array - ref_array
    return diff.tolist()

# UDF for blob detection and feature extraction
@udf(T.StructType([
    T.StructField("mean_diff", T.FloatType(), True),
    T.StructField("std_diff", T.FloatType(), True),
    T.StructField("hog_0", T.FloatType(), True),
    T.StructField("hog_1", T.FloatType(), True),
    T.StructField("hog_2", T.FloatType(), True),
    T.StructField("hog_3", T.FloatType(), True),
    T.StructField("hog_4", T.FloatType(), True),
    T.StructField("hog_5", T.FloatType(), True),
    T.StructField("hog_6", T.FloatType(), True),
    T.StructField("hog_7", T.FloatType(), True),
    T.StructField("hog_8", T.FloatType(), True),
    T.StructField("hog_9", T.FloatType(), True),
    T.StructField("hog_10", T.FloatType(), True),
    T.StructField("hog_11", T.FloatType(), True),
    T.StructField("label", T.IntegerType(), True)
]))
def extract_features_udf(diff_img_array, sci_img_array, threshold_sigma=5.0):
    """Detect blobs and extract features in a PySpark UDF"""
    if diff_img_array is None:
        return None
    
    try:
        diff_img = np.array(diff_img_array, dtype=np.float32)
        sci_img = np.array(sci_img_array, dtype=np.float32) if sci_img_array else None
        
        # Detect blobs
        med = np.median(diff_img)
        std = np.std(diff_img)
        threshold = med + threshold_sigma * std
        mask = diff_img > threshold
        lbl = label(mask)
        props = regionprops(lbl, intensity_image=diff_img)
        blobs = [p for p in props if p.area >= 3]
        
        if blobs:
            # Find blob closest to center
            h, w = diff_img.shape
            cx, cy = w / 2.0, h / 2.0
            best = min(blobs, key=lambda b: (b.centroid[0] - cy)**2 + (b.centroid[1] - cx)**2)
            
            # Extract features from blob
            y0, x0, y1, x1 = best.bbox
            patch_diff = diff_img[y0:y1, x0:x1]
            
            mean_patch = float(np.mean(patch_diff))
            std_patch = float(np.std(patch_diff))
            
            small = resize(patch_diff, (32, 32), anti_aliasing=True)
            hog_vec = hog(small, orientations=12, pixels_per_cell=(8, 8), 
                         cells_per_block=(1, 1), visualize=False, feature_vector=True)
            
            # Create feature dictionary
            features = {
                "mean_diff": mean_patch,
                "std_diff": std_patch,
                "label": 1
            }
            
            # Add HOG features
            for i in range(12):
                features[f"hog_{i}"] = float(hog_vec[i] if i < len(hog_vec) else 0.0)
            
            return features
        else:
            return None
            
    except Exception as e:
        return None

# UDF for negative sample feature extraction
@udf(T.StructType([
    T.StructField("mean_diff", T.FloatType(), True),
    T.StructField("std_diff", T.FloatType(), True),
    T.StructField("hog_0", T.FloatType(), True),
    T.StructField("hog_1", T.FloatType(), True),
    T.StructField("hog_2", T.FloatType(), True),
    T.StructField("hog_3", T.FloatType(), True),
    T.StructField("hog_4", T.FloatType(), True),
    T.StructField("hog_5", T.FloatType(), True),
    T.StructField("hog_6", T.FloatType(), True),
    T.StructField("hog_7", T.FloatType(), True),
    T.StructField("hog_8", T.FloatType(), True),
    T.StructField("hog_9", T.FloatType(), True),
    T.StructField("hog_10", T.FloatType(), True),
    T.StructField("hog_11", T.FloatType(), True),
    T.StructField("label", T.IntegerType(), True)
]))
def extract_negative_features_udf(diff_img_array, patch_size=32, neg_threshold_sigma=3.0, min_distance_from_center=12):
    """Extract features from negative samples in a PySpark UDF"""
    if diff_img_array is None:
        return None
    
    try:
        diff_img = np.array(diff_img_array, dtype=np.float32)
        h, w = diff_img.shape
        cx, cy = w / 2.0, h / 2.0
        
        patch_size = min(patch_size, h, w)
        
        # Calculate acceptance threshold
        global_med = np.median(diff_img)
        global_std = np.std(diff_img)
        accept_threshold = global_med + neg_threshold_sigma * global_std
        
        # Try to find a suitable negative patch
        for attempt in range(10):  # Reduced attempts for efficiency
            max_x = max(0, w - patch_size)
            max_y = max(0, h - patch_size)
            
            if max_x == 0 and max_y == 0:
                x0, y0 = 0, 0
            else:
                x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0
                y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            
            sub = diff_img[y0:y0 + patch_size, x0:x0 + patch_size]
            if sub.size == 0:
                continue
            
            px_cx = x0 + patch_size / 2.0
            px_cy = y0 + patch_size / 2.0
            dist_to_center = np.sqrt((px_cx - cx)**2 + (px_cy - cy)**2)
            
            if dist_to_center >= min_distance_from_center and np.max(sub) < accept_threshold:
                # Extract features
                mean_val = float(np.mean(sub))
                std_val = float(np.std(sub))
                
                sub_resized = resize(sub, (32, 32), anti_aliasing=True)
                hog_vec = hog(sub_resized, orientations=12, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=False, feature_vector=True)
                
                features = {
                    "mean_diff": mean_val,
                    "std_diff": std_val,
                    "label": 0
                }
                
                for i in range(12):
                    features[f"hog_{i}"] = float(hog_vec[i] if i < len(hog_vec) else 0.0)
                
                return features
        
        return None
        
    except Exception as e:
        return None