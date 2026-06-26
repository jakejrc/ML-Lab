# -*- coding: utf-8 -*-
import asyncio, requests, time
from nodriver import Browser, start

base = "http://127.0.0.1:7860/gradio_api/call/"

# Load data
r = requests.post(base + "on_load_data", json={"data": ["\u805a\u7c7b\u7403\u5f62 (Blobs)", 0.3]})
if r.status_code == 200:
    eid = r.json().get("event_id")
    time.sleep(5)
    requests.get(base + f"on_load_data/{eid}")
    print("data loaded")

# Train K-Means
r = requests.post(base + "on_train_clustering", json={"data": ["K-Means", 3, 300, 0.5, 5, "ward", "euclidean", "k-means++"]})
if r.status_code == 200:
    eid = r.json().get("event_id")
    time.sleep(6)
    requests.get(base + f"on_train_clustering/{eid}")
    print("kmeans trained")

async def main():
    browser = await start()
    page = await browser.get('http://127.0.0.1:7860')
    await asyncio.sleep(3)

    # Navigate to cluster via Gradio's internal radio click
    await page.evaluate('''() => {
        // Find all radio inputs
        const radios = document.querySelectorAll('input[type="radio"]');
        for (const r of radios) {
            // Check parent chain for text containing cluster
            let el = r;
            let found = false;
            while (el) {
                if (el.textContent && el.textContent.includes('\u805a\u7c7b\u5b9e\u9a8c')) {
                    found = true;
                    break;
                }
                el = el.parentElement;
            }
            if (found) {
                r.dispatchEvent(new Event('click', {bubbles: true}));
                r.dispatchEvent(new Event('change', {bubbles: true}));
                r.checked = true;
                break;
            }
        }
    }''')
    await asyncio.sleep(3)
    
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_trained.png')
    print("cluster trained screenshot saved")
    
    browser.stop()

asyncio.run(main())
