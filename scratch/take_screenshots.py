"""
Script para tirar screenshots das abas do dashboard.

Para rodar:
poetry run python scratch/take_screenshots.py

Requer: poetry install --with=dev

"""

import asyncio
import os
from playwright.async_api import async_playwright


PORTAL_URL = os.getenv("PORTAL_URL", "http://127.0.0.1:8000/dashboard")
VIEWPORT = {"width": 1440, "height": 900}
TABS = {
    "home": {"button": "#tab-home", "ready": "#mc-status"},
    "inference": {"button": "#tab-inference", "ready": "#payload"},
    "train": {"button": "#tab-train", "ready": "#runs-table-body"},
    "telemetry": {"button": "#tab-telemetry", "ready": "#resourceChart"},
}


async def main():
    assets_dir = os.path.abspath("assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()
        
        print(f"Acessando o Dashboard em {PORTAL_URL}...")
        await page.goto(PORTAL_URL, wait_until="domcontentloaded")
        await page.wait_for_selector("#tab-home")
        await page.wait_for_timeout(1000)

        for tab_name, tab in TABS.items():
            print(f"Clicando na aba {tab_name} ({tab['button']})...")
            await page.click(tab["button"])
            await page.wait_for_selector(tab["ready"])
            await page.wait_for_timeout(1200)
            await page.evaluate("window.scrollTo(0, 0)")
            
            screenshot_path = os.path.join(assets_dir, f"{tab_name}.png")
            await page.screenshot(path=screenshot_path)
            print(f"Screenshot salvo em {screenshot_path}")
            
        await browser.close()
        print("Tudo concluído com sucesso!")

if __name__ == "__main__":
    asyncio.run(main())
