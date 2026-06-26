# -*- coding: utf-8 -*-
import asyncio
from nodriver import Browser, start

async def main():
    browser = await start()
    page = await browser.get('http://127.0.0.1:7860')
    await asyncio.sleep(3)
    
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\home.png')
    print("homepage screenshot saved")
    
    radio_btns = await page.query_selector_all('input[type="radio"]')
    for btn in radio_btns:
        label = await btn.evaluate('el => el.parentElement?.textContent || ""')
        if '\u805a\u7c7b' in str(label):
            await btn.click()
            break
    await asyncio.sleep(2)
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_page.png')
    print("cluster page screenshot saved")
    
    browser.stop()

asyncio.run(main())
