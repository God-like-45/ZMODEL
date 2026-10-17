"""
CDP-based screen capture and interaction script for Z-ModelPilot Phase 5 Optimization UI.
Captures:
1. Empty state (no model loaded)
2. Configured setup state (linear_classifier.onnx loaded, local_onnx_runtime_cpu profile selected)
3. Executing / Completed optimization results with metrics, ranking table, and Pareto frontier.
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
CDP_PORT = 9222
PROFILE_DIR = r"d:\zmodel\temp_cdp_profile"


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


async def take_screenshot(ws, filename, clip=None):
    params = {"format": "png"}
    if clip:
        params["clip"] = clip
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


async def main():
    # 1. Start Edge headless
    print("Starting Edge in headless mode with remote debugging...")
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
    time.sleep(2.5)

    try:
        # 2. Get WebSocket debugger URL
        resp = requests.get(f"http://localhost:{CDP_PORT}/json/list")
        tabs = resp.json()
        print(f"Connected to Edge CDP. Open tabs: {len(tabs)}")
        ws_url = tabs[0]["webSocketDebuggerUrl"]

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

            # --- Step 1: Optimization View Empty State ---
            print("\nNavigating to Empty Optimization View: http://localhost:8501/?page=optimization")
            await send_cmd(ws, "Page.navigate", {"url": "http://localhost:8501/?page=optimization"})
            await asyncio.sleep(4.0)
            await take_screenshot(ws, "screenshot_p5_empty.png")

            # --- Step 2: Configured Optimization View ---
            url_configured = "http://localhost:8501/?page=optimization&sample=linear_classifier.onnx&profile=local_onnx_runtime_cpu"
            print(f"\nNavigating to Configured Setup View: {url_configured}")
            await send_cmd(ws, "Page.navigate", {"url": url_configured})
            await asyncio.sleep(4.0)
            await take_screenshot(ws, "screenshot_p5_configured.png")

            # --- Step 3: Trigger Optimization Run ---
            print("\nSearching for 'Run Optimization Analysis' button...")
            eval_res = await send_cmd(ws, "Runtime.evaluate", {
                "expression": """
                (() => {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const optBtn = buttons.find(b => b.innerText.includes('Run Optimization Search') || b.innerText.includes('Run Optimization'));
                    if (optBtn) {
                        optBtn.scrollIntoView();
                        optBtn.click();
                        return "CLICKED: " + optBtn.innerText;
                    }
                    return "NOT_FOUND. Buttons: " + buttons.map(b => b.innerText).join(', ');
                })()
                """
            })
            print(f"Button click result: {eval_res.get('result', {}).get('value')}")

            # Wait for optimization benchmarking to complete
            print("Waiting for optimization benchmarking execution to finish (up to 30s)...")
            for i in range(15):
                await asyncio.sleep(2.0)
                check_res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": """
                    (() => {
                        const text = document.body.innerText;
                        if (text.includes("Evaluated Candidate Configurations") || text.includes("Optimal Feasible") || text.includes("Download Report (JSON)")) {
                            return "COMPLETED";
                        }
                        if (text.includes("Optimization in progress") || text.includes("Benchmarking candidate")) {
                            return "PROGRESS";
                        }
                        return "RUNNING";
                    })()
                    """
                })
                status = check_res.get("result", {}).get("value")
                print(f"Poll {i+1}: {status}")
                if status == "COMPLETED":
                    break

            await asyncio.sleep(2.0)
            
            # 1. Get true scroll height
            height_res = await send_cmd(ws, "Runtime.evaluate", {
                "expression": """
                (() => {
                    const el = document.querySelector('[data-testid="stAppViewContainer"]') || document.body;
                    return Math.max(el.scrollHeight, document.documentElement.scrollHeight, 2400);
                })()
                """
            })
            content_height = int(height_res.get("result", {}).get("value", 2400))
            print(f"Detected true full scroll height: {content_height}px")

            # 2. Capture full-page results screenshot
            await send_cmd(ws, "Emulation.setDeviceMetricsOverride", {
                "width": 1400,
                "height": content_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            await asyncio.sleep(1.5)
            await take_screenshot(ws, "screenshot_p5_full_results.png")

            # 3. Reset viewport and scroll down to ranking & Pareto section for clear view
            await send_cmd(ws, "Emulation.setDeviceMetricsOverride", {
                "width": 1400,
                "height": 1050,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            await asyncio.sleep(0.5)
            await send_cmd(ws, "Runtime.evaluate", {
                "expression": """
                (() => {
                    const df = document.querySelector('[data-testid="stDataFrame"]') || document.querySelector('table');
                    if (df) {
                        df.scrollIntoView({ behavior: 'instant', block: 'center' });
                    }
                })()
                """
            })
            await asyncio.sleep(1.0)
            await take_screenshot(ws, "screenshot_p5_results_ranking.png")
            print("\nAll Phase 5 screenshots captured successfully!")

    finally:
        print("Terminating Edge process...")
        edge_proc.terminate()
        edge_proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
