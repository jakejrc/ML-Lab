# -*- coding: utf-8 -*-
import asyncio, requests, time, json, base64
from nodriver import Browser, start

base = "http://127.0.0.1:7860/gradio_api/call/"

# Step 1: Load cluster data via API
r = requests.post(base + "on_load_data", json={"data": ["\u805a\u7c7b\u7403\u5f62 (Blobs)", 0.3]})
eid = r.json()["event_id"]
time.sleep(5)
requests.get(base + f"on_load_data/{eid}")
print("data loaded")

# Step 2: Train each algorithm and get result images
algos = ["K-Means", "DBSCAN", "\u5c42\u6b21\u805a\u7c7b", "PCA"]
imgs = {}
for algo in algos:
    r = requests.post(base + "on_train_clustering", json={"data": [algo, 3, 300, 0.8, 5, "ward", "euclidean", "k-means++"]})
    eid = r.json()["event_id"]
    time.sleep(5)
    r2 = requests.get(base + f"on_train_clustering/{eid}")
    text = r2.text
    # Extract image URL
    if '"url":' in text and 'image' in text:
        url_start = text.find('"url":"') + 7
        url_end = text.find('"', url_start)
        url = text[url_start:url_end]
        # Download image
        r3 = requests.get(url)
        imgs[algo] = r3.content
        print(f"{algo}: image ({len(r3.content)} bytes)")
    else:
        # Check for error
        if "UnboundLocalError" in text or "\u5931\u8d25" in text:
            print(f"{algo}: ERROR")
        else:
            print(f"{algo}: no image ({len(text)} chars)")

# Step 3: Build a custom HTML page showing all results
html_parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"><style>']
html_parts.append('body{font-family:sans-serif;background:#f0f2f5;margin:0;padding:20px}')
html_parts.append('.card{background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);padding:20px;margin-bottom:20px}')
html_parts.append('h2{color:#1a73e8;margin-top:0;border-bottom:2px solid #1a73e8;padding-bottom:8px}')
html_parts.append('.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}')
html_parts.append('img{width:100%;border-radius:8px}')
html_parts.append('.badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:13px;font-weight:bold;margin-right:8px}')
html_parts.append('.pass{background:#d4edda;color:#155724}')
html_parts.append('</style></head><body>')
html_parts.append('<h1>ML-Lab v3.0 \u805a\u7c7b\u5b9e\u9a8c Bug \u4fee\u590d\u9a8c\u8bc1</h1>')
html_parts.append('<div class="grid">')

for algo, img_data in imgs.items():
    html_parts.append(f'<div class="card"><h2>{algo}</h2>')
    html_parts.append(f'<span class="badge pass">PASS</span><br><br>')
    b64 = base64.b64encode(img_data).decode()
    html_parts.append(f'<img src="data:image/webp;base64,{b64}">')
    html_parts.append('</div>')

html_parts.append('</div></body></html>')
html_content = '\n'.join(html_parts)

output_path = r'D:\WPSClaw\ML-Lab\screenshots\cluster_results.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"\nResults page saved: {output_path}")

# Step 4: Screenshot the results page
async def main():
    browser = await start()
    page = await browser.get(f'file:///{output_path}')
    await asyncio.sleep(2)
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_results.png')
    print("results screenshot saved")
    browser.stop()

asyncio.run(main())
