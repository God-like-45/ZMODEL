"""
CDP-based screen capture and verification script for Z-ModelPilot Phase 6 (Persistence & History).
Captures:
1. screenshot_p6_db_disconnected.png: History view with PostgreSQL disconnected / offline
"""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
import requests
import websockets

ARTIFACTS_DIR = r"C:\Users\Umesh Chandra\.gemini\antigravity-ide\brain\7c616fb4-c761-43e1-8be5-ad52a521b3cd"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CDP_PORT = 9228
PROFILE_DIR = r"d:\zmodel\temp_cdp_p6"


async def send_cmd(ws, method, params=None, msg_id=[1]):
    mid = msg_id[0]
    msg_id[0] += 1
    req = {"id": mid, "method": method, "params": params or {}}
    await ws.send(json.dumps(req))
    while True:
        resp_raw = await ws.recv()
        resp = json.loads(resp_raw)
        if resp.get("id") == mid:
            return resp.get("result", {})


async def take_screenshot(ws, filename):
    params = {"format": "png"}
    res = await send_cmd(ws, "Page.captureScreenshot", params)
    data = base64.b64decode(res["data"])
    workspace_path = os.path.join(r"d:\zmodel", filename)
    with open(workspace_path, "wb") as f:
        f.write(data)
    print(f"Saved workspace screenshot: {workspace_path} ({len(data)} bytes)")

    if os.path.exists(ARTIFACTS_DIR):
        artifact_path = os.path.join(ARTIFACTS_DIR, filename)
        shutil.copy2(workspace_path, artifact_path)
        print(f"Copied to artifact directory: {artifact_path}")
    return workspace_path


async def capture_disconnected():
    print("--- Capturing Test Case A: Disconnected DB ---")
    if os.path.exists(PROFILE_DIR):
        shutil.rmtree(PROFILE_DIR, ignore_errors=True)

    edge_proc = subprocess.Popen(
        [
            EDGE_PATH,
            "--headless=new",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={PROFILE_DIR}",
            "--window-size=1400,1050",
            "--hide-scrollbars",
            "--disable-gpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)

    try:
        new_tab = requests.put(f"http://localhost:{CDP_PORT}/json/new?http://localhost:8501/?page=history", timeout=5).json()
        ws_url = new_tab["webSocketDebuggerUrl"]
        print(f"Connected to new tab: {new_tab.get('id')}")

        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            await send_cmd(ws, "Page.enable")
            await send_cmd(ws, "DOM.enable")
            await send_cmd(ws, "Runtime.enable")
            await send_cmd(ws, "Emulation.setDeviceMetricsOverride", {
                "width": 1400,
                "height": 1050,
                "deviceScaleFactor": 1,
                "mobile": False,
            })

            print("Waiting for page load and banner...")
            for i in range(12):
                await asyncio.sleep(1.0)
                res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": "document.body.innerText.includes('PostgreSQL Database Unreachable') || document.body.innerText.includes('Disconnected')"
                })
                if res.get("result", {}).get("value"):
                    print(f"Banner detected at second {i+1}")
                    break

            await asyncio.sleep(1.0)
            await take_screenshot(ws, "screenshot_p6_db_disconnected.png")
            print("Successfully captured screenshot_p6_db_disconnected.png")
    finally:
        edge_proc.terminate()
        try:
            edge_proc.wait(timeout=3)
        except Exception:
            edge_proc.kill()


if __name__ == "__main__":
    asyncio.run(capture_disconnected())
