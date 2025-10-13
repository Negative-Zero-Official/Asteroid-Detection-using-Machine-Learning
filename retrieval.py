from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import udf
import tarfile
import fastavro
import io
import gzip
from astropy.io import fits
import numpy as np

# Initialize Spark Session
def get_spark_session():
    return SparkSession.builder \
        .appName("ZTF Alert Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

# Define schema for alerts
ALERT_SCHEMA = T.StructType([
    T.StructField("ra", T.DoubleType(), True),
    T.StructField("dec", T.DoubleType(), True),
    T.StructField("jd", T.DoubleType(), True),
    T.StructField("cutoutScience", T.BinaryType(), True),
    T.StructField("cutoutTemplate", T.BinaryType(), True),
    T.StructField("cutoutDifference", T.BinaryType(), True)
])

def parse_avro_alerts_from_tar_spark(tar_path, max_alerts=None):
    """Parse Avro alerts from tar file and return as Spark DataFrame"""
    spark = get_spark_session()
    
    def parse_tar_file(tar_path):
        """Generator function to parse tar file and yield alerts"""
        alerts = []
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
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
                        "ra": ra,
                        "dec": dec,
                        "jd": jd,
                        "cutoutScience": cs,
                        "cutoutTemplate": cr,
                        "cutoutDifference": cd,
                    })
                    
                    if max_alerts and len(alerts) >= max_alerts:
                        return alerts
        return alerts
    
    # Create RDD from parsed alerts and convert to DataFrame
    alerts_rdd = spark.sparkContext.parallelize(parse_tar_file(tar_path))
    alerts_df = spark.createDataFrame(alerts_rdd, ALERT_SCHEMA)
    
    return alerts_df

# UDF for decoding cutouts
@udf(T.ArrayType(T.ArrayType(T.FloatType())))
def decode_cutout_udf(stamp_bytes):
    """Decode cutout image from bytes to 2D array"""
    if stamp_bytes is None:
        return None
    try:
        decompressed = gzip.decompress(stamp_bytes)
        with fits.open(io.BytesIO(decompressed), memap=False) as hdul:
            arr = hdul[0].data.astype(np.float32)
        return arr.tolist()
    except Exception as e:
        return None

def process_alerts_with_spark(tar_paths, max_alerts_per_tar=None):
    """Process multiple tar files and return combined DataFrame"""
    spark = get_spark_session()
    
    all_dfs = []
    for tar_path in tar_paths:
        print(f"Processing tar file: {tar_path}")
        df = parse_avro_alerts_from_tar_spark(tar_path, max_alerts_per_tar)
        df = df.withColumn("science_array", decode_cutout_udf(F.col("cutoutScience")))
        df = df.withColumn("template_array", decode_cutout_udf(F.col("cutoutTemplate")))
        all_dfs.append(df)
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = all_dfs[0]
        for df in all_dfs[1:]:
            combined_df = combined_df.union(df)
        return combined_df
    else:
        return spark.createDataFrame([], ALERT_SCHEMA)