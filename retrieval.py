import tarfile
import fastavro
import io
import gzip
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

def parse_avro_alerts_from_tar(tar_path, max_alerts):
    alerts = []
    print(f"Processing tar file: {tar_path}")
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tqdm(tar.getmembers(), desc=f"Parsing AVRO alerts ({tar_path})", file=sys.stderr):
            if not member.isfile() or not member.name.endswith(".avro"):
                continue
            f = tar.extractfile(member)
            if not f:
                continue
            reader = fastavro.reader(f)
            for record in reader:
                cand = record.get("candidate", {})
                ra = cand.get("ra")
                dec = cand.get("dec")
                jd = cand.get("jd") or cand.get("jd_t")
                
                cs = record.get("cutoutScience", {}).get("stampData")
                cr = record.get("cutoutTemplate", {}).get("stampData")
                cd = record.get("cutoutDifference", {}).get("stampData")
                
                if ra is None or dec is None or cs is None:
                    continue

                alerts.append({
                    "ra" : ra,
                    "dec" : dec,
                    "jd" : jd,
                    "cutoutScience" : cs,
                    "cutoutTemplate" : cr,
                    "cutoutDifference" : cd,
                })
                
                if max_alerts and len(alerts) >= max_alerts:
                    return alerts
    return alerts

def decode_cutout(stamp_bytes: bytes):
    decompressed = gzip.decompress(stamp_bytes)
    with fits.open(io.BytesIO(decompressed), memap=False) as hdul:
        arr = hdul[0].data.astype(np.float32)
    return arr