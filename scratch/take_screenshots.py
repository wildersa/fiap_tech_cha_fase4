import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    assets_dir = os.path.abspath("assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Create context with a nice viewport size
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        
        print("Acessando o Dashboard...")
        await page.goto("http://127.0.0.1:8000/dashboard")
        await page.wait_for_timeout(2000) # wait for data load
        
        tabs = {
            "home": "#tab-home",
            "inference": "#tab-inference",
            "train": "#tab-train",
            "telemetry": "#tab-telemetry"
        }
        
        for tab_name, selector in tabs.items():
            print(f"Clicando na aba {tab_name} ({selector})...")
            await page.click(selector)
            await page.wait_for_timeout(1000) # wait for animations/rendering
            
            screenshot_path = os.path.join(assets_dir, f"{tab_name}.png")
            await page.screenshot(path=screenshot_path)
            print(f"Screenshot salvo em {screenshot_path}")
            
        await browser.close()
        print("Tudo concluído com sucesso!")

if __name__ == "__main__":
    asyncio.run(main())
