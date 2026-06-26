# -*- coding: utf-8 -*-
import asyncio, requests, time, json
from nodriver import Browser, start

base = "http://127.0.0.1:7860/gradio_api/call/"

async def main():
    browser = await start()
    page = await browser.get('http://127.0.0.1:7860')
    await asyncio.sleep(4)

    # Get Gradio session hash from cookies
    cookies = await browser.cookies.get_all()
    session_hash = None
    for c in cookies:
        if c.name == '_gradio_session':
            session_hash = c.value
            break
    
    if not session_hash:
        # Try to get from page JS
        session_hash = await page.evaluate('''() => {
            try { return window.gradio_config?.session_hash; } catch(e) { return null; }
        }''')
    
    print(f"session_hash: {session_hash}")
    
    # Use Gradio's internal set_component_value to switch page
    await page.evaluate('''() => {
        // Find Gradio radio component for nav
        const inputs = document.querySelectorAll('input[type="radio"]');
        for (const inp of inputs) {
            let el = inp.closest('.gradio-radio') || inp.parentElement?.parentElement;
            if (!el) continue;
            // Check all spans for cluster text
            const spans = el.querySelectorAll('span, label, div');
            let isCluster = false;
            for (const s of spans) {
                if (s.textContent.includes('\u805a\u7c7b\u5b9e\u9a8c')) {
                    isCluster = true;
                    break;
                }
            }
            if (isCluster) {
                inp.click();
                console.log('clicked cluster radio');
                return true;
            }
        }
        return false;
    }''')
    await asyncio.sleep(4)
    
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_click.png')
    print("cluster click screenshot saved")
    
    # If still on data page, try clicking via nav-tag in header
    await page.evaluate('''() => {
        const tags = document.querySelectorAll('.nav-tag, [data-nav]');
        for (const t of tags) {
            if (t.dataset.nav === '\u805a\u7c7b\u5b9e\u9a8c' || t.textContent.includes('\u805a\u7c7b\u5b9e\u9a8c')) {
                t.click();
                return true;
            }
        }
        return false;
    }''')
    await asyncio.sleep(4)
    
    await page.save_screenshot(r'D:\WPSClaw\ML-Lab\screenshots\cluster_navtag.png')
    print("cluster navtag screenshot saved")
    
    browser.stop()

asyncio.run(main())
