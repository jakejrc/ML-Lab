# -*- coding: utf-8 -*-
import asyncio, requests, time
from nodriver import Browser, start

# Load cluster dataset via API first
base = "http://127.0.0.1:7860/gradio_api/call/"
r = requests.post(base + "on_load_data", json={"data": ["\u805a\u7c7b\u7403\u5f62 (Blobs)", 0.3]})
if r.status_code == 200:
    eid = r.json().get("event_id")
    if eid:
        time.sleep(5)
        requests.get(base + f"on_load_data/{eid}")
        print("data loaded via API")

async def main():
    browser = await start()
    page = await browser.get('http://127.0.0.1:7860')
    await asyncio.sleep(3)

    # Click the cluster radio button by evaluating JS
    await page.evaluate('''() => {
        const radios = document.querySelectorAll('input[type="radio"]');
        for (const r of radios) {
            const lbl = r.closest('label') || r.parentElement;
            if (lbl && lbl.textContent.includes('\u805a\u7c7b')) {
                r.click();
                break;
            }
        }
    }''')
    await asyncio.sleep(2)
    
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_page.png')
    print("cluster page screenshot saved")
    
    browser.stop()

asyncio.run(main())
