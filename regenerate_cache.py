#!/usr/bin/env python
"""
Regenerate demo cache from latest trained models.
Run this after retraining models on bigger dataset.
"""
import json
import requests
import sys
from pathlib import Path

SAMPLE_IMAGES = [
    "frontend/static/sample_images/sample_1.jpg",
    "frontend/static/sample_images/sample_2.jpg",
    "frontend/static/sample_images/sample_3.jpg",
    "frontend/static/sample_images/sample_4.jpg",
    "frontend/static/sample_images/sample_5.jpg",
    "frontend/static/sample_images/sample_6.jpg",
]

def regenerate_cache(server_url="http://localhost:5000", demo_mode_off=False):
    """Regenerate cache by making predictions on sample images."""
    cache = {}

    print("🔄 Regenerating demo cache from live predictions...")
    print(f"Server: {server_url}")
    print(f"Demo Mode OFF: {demo_mode_off}")
    print()

    for idx, img_path in enumerate(SAMPLE_IMAGES, 1):
        key = f"sample_{idx}"

        if not Path(img_path).exists():
            print(f"⚠️  {key}: Image not found at {img_path}")
            continue

        try:
            # Make prediction
            with open(img_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f'{server_url}/predict', files=files, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    cache[key] = {
                        'image_path': f'sample_{idx}.jpg',
                        'vehicle_count': int(data['vehicle_count']),
                        'car_count': int(data['car_count']),
                        'motorcycle_count': int(data['motorcycle_count']),
                        'traffic_label': data['traffic_label'],
                        'predictions': {
                            'random_forest': data['predictions'].get('random_forest', 'Low'),
                            'decision_tree': data['predictions'].get('decision_tree', 'Low'),
                            'logistic_regression': data['predictions'].get('logistic_regression', 'Low'),
                            'mlp': data['predictions'].get('mlp', 'Low'),
                            'svm': data['predictions'].get('svm', 'Low'),
                            'knn': data['predictions'].get('knn', 'Low')
                        },
                        'linear_regression': float(data['predictions'].get('linear_regression', 0)),
                        'features': {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in data['features'].items()}
                    }
                    print(f"✓ {key}: {data['vehicle_count']} vehicles, {data['traffic_label']}")
                else:
                    print(f"✗ {key}: {data.get('error', 'Unknown error')}")
            else:
                print(f"✗ {key}: HTTP {response.status_code}")
        except Exception as e:
            print(f"✗ {key}: {str(e)}")

    # Save cache
    cache_path = Path('frontend/static/sample_images/sample_cache.json')
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\n✓ Cache regenerated: {len(cache)} samples saved to {cache_path}")
    print("\n📌 Remember to restart the Flask server to load new cache!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Regenerate demo cache')
    parser.add_argument('--server', default='http://localhost:5000', help='Flask server URL')
    parser.add_argument('--live', action='store_true', help='Use live predictions (not demo mode)')
    args = parser.parse_args()

    regenerate_cache(args.server, demo_mode_off=args.live)
