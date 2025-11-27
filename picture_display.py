from retrieval import parse_avro_alerts_from_tar, decode_cutout
from preprocessing import preprocess_image, compute_difference
import matplotlib.pyplot as plt

tar = "tarballs\\ztf_public_20250819.tar.gz"

alerts = parse_avro_alerts_from_tar(tar, 5)

for a in alerts:
    sci = a.get('cutoutScience')
    ref = a.get('cutoutTemplate')
    diff = a.get('cutoutDifference')
    
    sci_dec = decode_cutout(sci)
    ref_dec = decode_cutout(ref) if ref is not None else None
    diff = decode_cutout(diff)
    
    sci_pre = preprocess_image(sci_dec)
    ref_pre = preprocess_image(ref_dec) if ref is not None else None
    diff_ours = compute_difference(sci_dec, ref_dec)


    if sci_dec is None:
        # nothing to display for this alert
        continue

    if ref_dec is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0,0].imshow(sci_dec, cmap='gray')
        axes[0,0].set_title('Science')
        axes[0,0].axis('off')
        axes[0,1].imshow(ref_dec, cmap='gray')
        axes[0,1].set_title('Template')
        axes[0,1].axis('off')
        axes[0,2].imshow(sci_pre, cmap='gray')
        axes[0,2].set_title('Science Preprocessed')
        axes[0,2].axis('off')
        axes[1,0].imshow(ref_pre, cmap='gray')
        axes[1,0].set_title('Reference Preprocessed')
        axes[1,0].axis('off')
        axes[1,1].imshow(diff, cmap='gray')
        axes[1,1].set_title('Difference')
        axes[1,1].axis('off')
        axes[1,2].imshow(diff_ours, cmap='gray')
        axes[1,2].set_title('Difference Ours')
        axes[1,2].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(sci_dec, cmap='gray')
        ax.set_title('Science')
        ax.axis('off')

    plt.tight_layout()
    plt.show()