from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=50)  # slow_mo makes it easier to see
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    page.goto("https://example.com")

    # Move mouse and click somewhere
    page.mouse.move(200, 200)
    time.sleep(2)
    page.mouse.click(200, 200)

    # Or click an element (recommended if you have selectors)
    page.click("text=More information")

    time.sleep(5)
    browser.close()
