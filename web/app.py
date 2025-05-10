import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64
from mimetypes import guess_type
from datetime import datetime
import boto3
import uuid
from concurrent.futures import ThreadPoolExecutor

# Initialize Thread Pool
executor = ThreadPoolExecutor(max_workers=2)

# MinIO setup
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'
)

app = Flask(__name__)

# Ensure the uploads folder exists
os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

# FastAPI server URL (env var)
FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']

# === UPDATED: Helper function for getting object key ===
def get_object_key(pred_font, prediction_id, filename):
    """
    Create S3 object key like: font_Arial/<uuid>.png
    """
    safe_font_name = pred_font.replace(" ", "_")  # e.g., Times New Roman -> Times_New_Roman
    font_dir = f"font_{safe_font_name}"
    ext = os.path.splitext(filename)[1]
    return f"{font_dir}/{prediction_id}{ext}"

# === Upload to MinIO bucket ===
def upload_to_bucket(img_path, pred_font, confidence, prediction_id):
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    bucket_name = "fontdetector-uploads"  # updated bucket name (adjust as needed)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = get_object_key(pred_font, prediction_id, img_path)
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
        )

    # Tag the object with predicted font + confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_font', 'Value': pred_font},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )

# === Request inference from FastAPI ===
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()
        
        result = response.json()
        predicted_font = result.get("prediction")
        probability = result.get("probability")
        
        return predicted_font, probability

    except Exception as e:
        print(f"Error during inference: {e}")  
        return None, None  

# === Routes ===
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    pred_font = None
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.instance_path, 'uploads', filename)
        f.save(save_path)

        # create a unique prediction ID
        prediction_id = str(uuid.uuid4())
        
        pred_font, prob = request_fastapi(save_path)
        if pred_font:
            executor.submit(upload_to_bucket, save_path, pred_font, prob, prediction_id)
            
            # Build flag icon form
            s3_key = get_object_key(pred_font, prediction_id, filename)
            flag_icon = f'''
                <form method="POST" action="/flag/{s3_key}" style="display:inline">
                    <button type="submit" class="btn btn-outline-warning btn-sm">🚩</button>
                </form>'''
            return f'<button type="button" class="btn btn-info btn-sm">{pred_font}</button> {flag_icon}'

    return '<a href="#" class="badge badge-warning">Warning: Prediction failed</a>'

@app.route('/flag/<path:key>', methods=['POST'])
def flag_object(key):
    bucket = "fontdetector-uploads"
    current_tags = s3.get_object_tagging(Bucket=bucket, Key=key)['TagSet']
    tags = {t['Key']: t['Value'] for t in current_tags}

    if "flagged" not in tags:
        tags["flagged"] = "true"
        tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
        s3.put_object_tagging(Bucket=bucket, Key=key, Tagging={'TagSet': tag_set})

    return '', 204  # No Content: stay on same page

@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    pred_font, prob = request_fastapi(img_path)
    return f"Predicted font: {pred_font}, Confidence: {prob:.2f}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
