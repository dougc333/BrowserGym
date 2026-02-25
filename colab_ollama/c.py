from playwright.sync_api import sync_playwright
import time

CURSOR_AND_BUTTON_JS = r"""
(() => {
  // --- visible cursor overlay ---
  if (!window.__pwCursor) {
    const c = document.createElement('div');
    c.id = '__pw_cursor';
    c.style.position = 'fixed';
    c.style.left = '0px';
    c.style.top = '0px';
    c.style.width = '14px';
    c.style.height = '14px';
    c.style.border = '2px solid red';
    c.style.borderRadius = '50%';
    c.style.background = 'rgba(255,0,0,0.15)';
    c.style.zIndex = '2147483647';
    c.style.pointerEvents = 'none';
    c.style.transition = 'transform 0.02s linear';
    document.documentElement.appendChild(c);
    window.__pwCursor = c;
    window.__pwMoveCursor = (x, y) => { c.style.transform = `translate(${x}px, ${y}px)`; };
  }

  // --- demo button ---
  if (!document.getElementById('__pw_demo_btn')) {
    const btn = document.createElement('button');
    btn.id = '__pw_demo_btn';
    btn.textContent = 'Click me';
    btn.style.position = 'fixed';
    btn.style.left = '50%';
    btn.style.top = '40%';
    btn.style.transform = 'translate(-50%, -50%)';
    btn.style.padding = '14px 18px';
    btn.style.fontSize = '18px';
    btn.style.borderRadius = '12px';
    btn.style.border = '1px solid #999';
    btn.style.background = '#f7f7f7';
    btn.style.cursor = 'pointer';
    btn.style.zIndex = '2147483646';

    btn.addEventListener('click', () => {
      btn.textContent = '✅ Clicked!';
      btn.style.background = '#d1fae5';
      btn.style.borderColor = '#10b981';
    });

    document.body.appendChild(btn);
  }
})();
"""

with sync_playwright() as p:
    browser = p.webkit.launch(headless=False, slow_mo=30)
    page = browser.new_page(viewport={"width": 1200, "height": 800})

    page.goto("http://127.0.0.1:8000/miniwob/click-checkboxes-transfer.html")
    page.wait_for_load_state()

    # Inject cursor+button into the page
    page.evaluate(CURSOR_AND_BUTTON_JS)

    # Get button center (for accurate clicking)
    btn = page.locator("#__pw_demo_btn")
    box = btn.bounding_box()
    assert box is not None
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2

    # Move cursor overlay along a path toward the button
    path = [(100, 120), (250, 200), (400, 260), (cx, cy)]
    for x, y in path:
        page.mouse.move(x, y)
        page.evaluate("([x,y]) => window.__pwMoveCursor(x,y)", [x, y])
        time.sleep(0.3)

    # Click the button
    page.mouse.click(cx, cy)
    page.evaluate("([x,y]) => window.__pwMoveCursor(x,y)", [cx, cy])

    time.sleep(20)
    browser.close()
