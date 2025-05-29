import boto3
import csv
import json
from datetime import datetime
import os

def lambda_handler(event, context):
    # Initialize clients
    bedrock = boto3.client('bedrock-runtime')
    s3 = boto3.client('s3')
    
    # Define models to test
    models = [
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
        'meta.llama3-70b-instruct-v1:0',
        'mistral.mistral-7b-instruct-v0:2'
    ]

    prompt = "What is a cosmetic guidance?"
    results = []

    for model_id in models:
        try:
            # Format request based on model type
            if 'anthropic' in model_id:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                })
            elif 'meta.llama' in model_id:
                body = json.dumps({
                    "prompt": f"<s>[INST] {prompt} [/INST]",
                    "max_gen_len": 1000,
                    "temperature": 0.7
                })
            elif 'mistral' in model_id:
                body = json.dumps({
                    "prompt": f"<s>[INST] {prompt} [/INST]",
                    "max_tokens": 1000,
                    "temperature": 0.7
                })
            
            response = bedrock.invoke_model(
                modelId=model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text based on model response format
            if 'anthropic' in model_id:
                output_text = response_body['content'][0]['text']
            elif 'meta.llama' in model_id:
                output_text = response_body['generation']
            elif 'mistral' in model_id:
                output_text = response_body['outputs'][0]['text']
            
            results.append({
                'model': model_id,
                'prompt': prompt,
                'response': output_text,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            results.append({
                'model': model_id,
                'prompt': prompt,
                'response': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })

    # Write to /tmp directory (writable in Lambda)
    tmp_file_path = '/tmp/ai_comparison.csv'
    with open(tmp_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['model', 'prompt', 'response', 'timestamp'])
        writer.writeheader()
        writer.writerows(results)
    
    # Upload to S3
    bucket_name = 'awstestbedrockaidrt'  # Replace with your bucket
    s3_key = f'ai-comparisons/comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    s3.upload_file(tmp_file_path, bucket_name, s3_key)
    
    # Clean up temp file
    os.remove(tmp_file_path)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Comparison completed and saved to s3://{bucket_name}/{s3_key}',
            'results_count': len(results)
        })
    }