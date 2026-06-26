# -*- coding: utf-8 -*-
import asyncio
from nodriver import Browser, start

async def main():
    browser = await start()
    page = await browser.get(f'file:///D:/WPSClaw/ML-Lab/screenshots/cluster_showcase.html')
    await asyncio.sleep(2)
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_showcase.png')
    print("showcase screenshot saved")
    browser.stop()

asyncio.run(main())
