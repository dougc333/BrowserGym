# Claude-as-controller demo for BrowserGym.
# Run these from this folder (claude_controller_demo/).

MINIWOB_DIR  ?= $(abspath ../miniwob-plusplus)
export MINIWOB_URL ?= file://$(MINIWOB_DIR)/miniwob/html/miniwob/

.PHONY: install miniwob dev serve build check help

help:
	@echo "make install  - pip install BrowserGym (core + miniwob) + bridge deps, npm install, playwright chromium"
	@echo "make miniwob  - clone MiniWoB++ next to the repo (uses the repo's setup-miniwob target)"
	@echo "make dev      - bridge on :8000 + Vite dev server on :5173 (hot reload)"
	@echo "make serve    - build the front end and serve everything from :8000"
	@echo "make check    - typecheck the front end"

install:
	pip install -e ../browsergym/core -e ../browsergym/miniwob
	pip install -r server/requirements.txt
	playwright install chromium
	cd web && npm install

miniwob:
	$(MAKE) -C .. setup-miniwob

dev:
	@echo "MINIWOB_URL=$(MINIWOB_URL)"
	@trap 'kill 0' INT TERM EXIT; \
	(cd server && uvicorn app:app --reload --port 8000) & \
	(cd web && npm run dev) & \
	wait

build:
	cd web && npm run build

serve: build
	@echo "Open http://localhost:8000"
	cd server && uvicorn app:app --port 8000

check:
	cd web && npm run typecheck
