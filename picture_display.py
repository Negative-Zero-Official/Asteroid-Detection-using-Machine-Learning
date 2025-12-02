from retrieval import parse_avro_alerts_from_tar, decode_cutout
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

    if sci_dec is None:
        # nothing to display for this alert
        continue
    
    for key, item in a.items():
        if key not in ['cutoutScience', 'cutoutTemplate', 'cutoutDifference']:
            print(f"{key+":":<15} {a[key]}")
    print("\n\n")
    
    if ref_dec is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(sci_dec, cmap='gray')
        axes[0].set_title('Science')
        axes[0].axis('off')
        axes[1].imshow(ref_dec, cmap='gray')
        axes[1].set_title('Template')
        axes[1].axis('off')
        axes[2].imshow(diff, cmap='gray')
        axes[2].set_title('Difference')
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(sci_dec, cmap='gray')
        ax.set_title('Science')
        ax.axis('off')

    plt.tight_layout()
    plt.show()