"""
Capture Phase 6 History Workspace Tabs using Chrome DevTools Protocol (CDP).
Captures:
1. screenshot_p6_history_experiments.png
2. screenshot_p6_history_comparison.png
3. screenshot_p6_history_models.png
4. screenshot_p6_history_diagnostics.png
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
CDP_PORT = 9230
PROFILE_DIR = r"d:\zmodel\temp_cdp_p6_tabs"


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


async def take_screenshot(ws, filename, max_height=1600):
    # Determine full height
    h_res = await send_cmd(ws, "Runtime.evaluate", {
        "expression": """
        (() => {
            const el = document.querySelector('[data-testid="stAppViewContainer"]') || document.body;
            return Math.min(Math.max(el.scrollHeight, document.documentElement.scrollHeight, 1050), 2200);
        })()
        """
    })
    target_height = int(h_res.get("result", {}).get("value", 1050))
    target_height = min(target_height, max_height)

    await send_cmd(ws, "Emulation.setDeviceMetricsOverride", {
        "width": 1400,
        "height": target_height,
        "deviceScaleFactor": 1,
        "mobile": False,
    })
    await asyncio.sleep(0.5)

    res = await send_cmd(ws, "Page.captureScreenshot", {"format": "png"})
    data = base64.b64decode(res["data"])
    workspace_path = os.path.join(r"d:\zmodel", filename)
    with open(workspace_path, "wb") as f:
        f.write(data)
    print(f"Saved: {workspace_path} ({len(data)} bytes, {target_height}px high)")

    if os.path.exists(ARTIFACTS_DIR):
        artifact_path = os.path.join(ARTIFACTS_DIR, filename)
        shutil.copy2(workspace_path, artifact_path)
        print(f"Copied to artifacts: {artifact_path}")
    return workspace_path


async def click_tab_by_text(ws, tab_text_fragment):
    eval_res = await send_cmd(ws, "Runtime.evaluate", {
        "expression": f"""
        (() => {{
            const tabs = Array.from(document.querySelectorAll('[role="tab"], button[data-testid="stTab"], [data-baseweb="tab"]'));
            const target = tabs.find(t => t.innerText && t.innerText.includes('{tab_text_fragment}'));
            if (target) {{
                target.scrollIntoView();
                target.click();
                return "CLICKED: " + target.innerText;
            }}
            return "NOT_FOUND. Available: " + tabs.map(t => t.innerText).join(' | ');
        }})()
        """
    })
    val = eval_res.get("result", {}).get("value")
    safe_val = str(val).encode("ascii", "replace").decode("ascii") if val else ""
    print(f"Tab click [{tab_text_fragment}]: {safe_val}")
    return val


async def main():
    print("Starting Edge in headless mode with remote debugging on port", CDP_PORT)
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

            print("Waiting for History page tabs to be present...")
            for i in range(15):
                await asyncio.sleep(1.0)
                res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": "document.querySelectorAll('[role=\"tab\"]').length >= 4"
                })
                if res.get("result", {}).get("value"):
                    print(f"Tabs detected at second {i+1}")
                    break

            await asyncio.sleep(2.0)

            # 1. Tab 1: Optimization Experiments (default active)
            print("\nCapturing Tab 1: Optimization Experiments...")
            await click_tab_by_text(ws, "Optimization Experiments")
            await asyncio.sleep(2.0)
            await take_screenshot(ws, "screenshot_p6_history_experiments.png", max_height=1800)

            # 2. Tab 2: Benchmark Runs & Comparison
            print("\nCapturing Tab 2: Benchmark Runs & Comparison...")
            await click_tab_by_text(ws, "Benchmark Runs")
            await asyncio.sleep(2.5)
            await take_screenshot(ws, "screenshot_p6_history_comparison.png", max_height=1800)

            # 3. Tab 3: Model Registry
            print("\nCapturing Tab 3: Model Registry...")
            await click_tab_by_text(ws, "Model Registry")
            await asyncio.sleep(2.0)
            await take_screenshot(ws, "screenshot_p6_history_models.png", max_height=1500)

            # 4. Tab 4: Database & Migrations
            print("\nCapturing Tab 4: Database & Migrations...")
            await click_tab_by_text(ws, "Database & Migrations")
            await asyncio.sleep(2.0)
            await take_screenshot(ws, "screenshot_p6_history_diagnostics.png", max_height=1600)

            print("\nAll Phase 6 screenshots successfully captured!")

    finally:
        edge_proc.terminate()
        try:
            edge_proc.wait(timeout=3)
        except Exception:
            edge_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
