# -*- coding: utf-8 -*-
import asyncio, requests, time, json, base64, os
from nodriver import Browser, start

base = "http://127.0.0.1:7860/gradio_api/call/"
out_dir = r"D:\WPSClaw\ML-Lab\screenshots"
os.makedirs(out_dir, exist_ok=True)

# Step 1: Load data
print("Loading data...")
r = requests.post(base + "on_load_data", json={"data": ["\u805a\u7c7b\u7403\u5f62 (Blobs)", 0.3]}, timeout=10)
if r.status_code != 200:
    print(f"Load failed: {r.status_code}")
    exit(1)
eid = r.json()["event_id"]
time.sleep(4)
r2 = requests.get(base + f"on_load_data/{eid}", timeout=10)
print(f"Load result: {r2.status_code}, {len(r2.text)} chars")

# Step 2: Train algorithms one by one
algos = ["K-Means", "DBSCAN", "\u5c42\u6b21\u805a\u7c7b", "PCA"]
for algo in algos:
    print(f"\nTraining {algo}...")
    try:
        r = requests.post(base + "on_train_clustering", 
            json={"data": [algo, 3, 300, 0.8, 5, "ward", "euclidean", "k-means++"]}, 
            timeout=15)
        if r.status_code != 200:
            print(f"  POST failed: {r.status_code}")
            continue
        eid = r.json().get("event_id")
        if not eid:
            print(f"  No event_id: {r.text[:100]}")
            continue
        time.sleep(6)
        r2 = requests.get(base + f"on_train_clustering/{eid}", timeout=15)
        text = r2.text
        
        # Parse SSE response
        for line in text.split('\n'):
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[null]' or data_str.strip() == 'null':
                    continue
                try:
                    data = json.loads(data_str)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'url' in item:
                                url = item['url']
                                # Fix URL encoding
                                url = url.replace('\\u003c', '<').replace('\\u003e', '>')
                                img_r = requests.get(url, timeout=10)
                                fname = f"cluster_{algo.replace(' ', '_')}.png"
                                fpath = os.path.join(out_dir, fname)
                                with open(fpath, 'wb') as f:
                                    f.write(img_r.content)
                                print(f"  Saved: {fname} ({len(img_r.content)} bytes)")
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"  Error: {e}")

# List saved images
print("\n=== Saved images ===")
for f in sorted(os.listdir(out_dir)):
    if f.startswith('cluster_') and f.endswith('.png'):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f}: {size:,} bytes")

print("\nDone!")
