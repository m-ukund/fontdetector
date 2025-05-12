import boto3
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class CanaryAnalyzer:
    def __init__(self):
        # Initialize MinIO client
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.environ.get('MINIO_URL', 'http://minio:9000'),
            aws_access_key_id=os.environ.get('MINIO_USER', 'your-access-key'),
            aws_secret_access_key=os.environ.get('MINIO_PASSWORD', 'your-secret-key'),
            region_name='us-east-1'
        )
        
    def get_prediction_data(self, days: int = 7) -> pd.DataFrame:
        """Get prediction data from MinIO for the last N days"""
        start_time = datetime.now() - timedelta(days=days)
        
        # Get data from both buckets
        prod_data = self._get_bucket_data("production-data", start_time)
        canary_data = self._get_bucket_data("canary-data", start_time)
        
        # Combine data
        all_data = prod_data + canary_data
        return pd.DataFrame(all_data)
    
    def _get_bucket_data(self, bucket: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get data from a specific bucket"""
        data = []
        try:
            objects = self.s3.list_objects_v2(Bucket=bucket)
            for obj in objects.get('Contents', []):
                if obj['LastModified'] >= start_time:
                    response = self.s3.get_object(Bucket=bucket, Key=obj['Key'])
                    content = response['Body'].read().decode('utf-8')
                    data.append(json.loads(content))
        except Exception as e:
            print(f"Error getting data from {bucket}: {e}")
        return data
    
    def generate_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive report of canary vs production performance"""
        df = self.get_prediction_data(days)
        
        # Calculate metrics
        metrics = {
            "total_requests": {
                "production": len(df[~df['is_canary']]),
                "canary": len(df[df['is_canary']])
            },
            "accuracy": {
                "production": df[~df['is_canary']]['is_correct'].mean(),
                "canary": df[df['is_canary']]['is_correct'].mean()
            },
            "avg_latency": {
                "production": df[~df['is_canary']]['latency'].mean(),
                "canary": df[df['is_canary']]['latency'].mean()
            },
            "feedback_rate": {
                "production": df[~df['is_canary']]['has_feedback'].mean(),
                "canary": df[df['is_canary']]['has_feedback'].mean()
            }
        }
        
        # Generate visualizations
        self._generate_visualizations(df, metrics)
        
        return metrics
    
    def _generate_visualizations(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Generate visualizations for the report"""
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.groupby(['timestamp', 'is_canary'])['is_correct'].mean().unstack().plot(
            ax=axes[0, 0], title='Accuracy Over Time'
        )
        
        # Latency distribution
        sns.boxplot(x='is_canary', y='latency', data=df, ax=axes[0, 1])
        axes[0, 1].set_title('Latency Distribution')
        
        # Request volume
        df.groupby(['timestamp', 'is_canary']).size().unstack().plot(
            ax=axes[1, 0], title='Request Volume Over Time'
        )
        
        # Feedback rate
        df.groupby(['timestamp', 'is_canary'])['has_feedback'].mean().unstack().plot(
            ax=axes[1, 1], title='Feedback Rate Over Time'
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig('canary_analysis_report.png')
        plt.close()

if __name__ == "__main__":
    analyzer = CanaryAnalyzer()
    report = analyzer.generate_report()
    print("Canary Analysis Report:")
    print(json.dumps(report, indent=2)) 