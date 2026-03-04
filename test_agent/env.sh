
# Prevents Chromium from using too much shared memory
export BROWSERGYM_CHROME_ARGS="--disable-dev-shm-usage --no-sandbox"
# Run headless to save GPU/Display memory
export BROWSERGYM_HEADLESS=1
export MINIWOB_URL="http://localhost:8000/miniwob/" 
