import json
from google.cloud import storage
from datetime import datetime


def get_client():
    return storage.Client()


def upload_json_to_gcs(bucket_name: str, blob_name: str, data: dict):
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(
        json.dumps(data),
        content_type="application/json"
    )


def download_json_from_gcs(bucket_name: str, blob_name: str):
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    content = blob.download_as_text()
    return json.loads(content)

def find_prediction_in_bucket(bucket_name: str, prediction_id: str):
    client = get_client()
    bucket = client.bucket(bucket_name)

    blobs = client.list_blobs(bucket_name, prefix="predictions/")

    for blob in blobs:
        if prediction_id in blob.name:
            content = blob.download_as_text()
            return json.loads(content)

    return None

def upload_file_to_gcs(bucket_name: str, local_path: str, blob_name: str):
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    
def get_latest_blob(bucket_name: str, prefix: str) -> str | None:
    client = get_client()
    
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))

    if not blobs:
        return None

    # tri par date de modification
    latest_blob = max(blobs, key=lambda b: b.updated)

    return latest_blob.name

def download_blob_to_file(bucket_name: str, blob_name: str, destination_path: str):
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(destination_path)