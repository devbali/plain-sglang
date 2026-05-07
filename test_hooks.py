#!/usr/bin/env python3
"""Test script: verify all 4 scheduling hooks fire with two different uids."""

import subprocess
import sys
import time
import requests
import json
import os
import signal
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 30000
BASE_URL = f"http://localhost:{PORT}/v1"

# Path to sglang repo on VM
REPO = Path.home() / "fairinf-sglang-upstream"


def start_server():
    """Start SGLang server with logging policy."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "python")
    
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--disable-cuda-graph",
        "--mem-fraction-static", "0.3",
        "--scheduling-policy-path",
        "sglang.srt.scheduling_hooks.logging_policy.LoggingSchedulingPolicy",
    ]
    
    print(f"[TEST] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc


def wait_for_server(proc, timeout=300):
    """Wait for the server to be ready."""
    print("[TEST] Waiting for server to be ready...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            print(f"\n[TEST] Server exited early! Output:\n{stdout[-2000:]}")
            sys.exit(1)
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(" ready!")
                return
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print("\n[TEST] Timeout waiting for server!")
    sys.exit(1)


def send_chat_request(user_id: str, message: str, port=PORT):
    """Send a chat completion request with a specific user ID."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": message}],
        "max_completion_tokens": 50,
        "temperature": 0.0,
        "user": user_id,
    }
    r = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    proc = None
    try:
        proc = start_server()
        wait_for_server(proc)

        print("\n[TEST] Sending request for user-alice...")
        alice = send_chat_request("user-alice", "What is 2+2? Answer in one word.")
        print(f"  Alice: {alice['choices'][0]['message']['content'][:100]}")

        print("[TEST] Sending request for user-bob...")
        bob = send_chat_request("user-bob", "What is the capital of France? One word.")
        print(f"  Bob: {bob['choices'][0]['message']['content'][:100]}")

        # Wait a moment for hooks to fire
        time.sleep(5)

        print("\n[TEST] Checking hook results (scanning stdout)...")
        
        # Gather server output
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        stdout = proc.stdout.read() if proc.stdout else ""

        # Check for hook output
        hook_lines = [l for l in stdout.split("\n") if "[HOOK]" in l]
        print(f"\nFound {len(hook_lines)} hook invocation lines:")
        for line in hook_lines:
            print(f"  {line}")

        # Validate
        checks = {
            "on_new_request": 0,
            "on_prefill_decision": 0,
            "on_decode_decision": 0,
            "on_end_of_scheduler_pass": 0,
        }
        for line in hook_lines:
            for key in checks:
                if key in line:
                    checks[key] += 1

        print("\n" + "=" * 50)
        print("HOOK TEST RESULTS")
        print("=" * 50)
        
        all_fired = True
        for hook, count in checks.items():
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {hook}: {count} call(s)")
            if count == 0:
                all_fired = False
        
        # Check UIDs were observed
        uids_seen = set()
        for line in hook_lines:
            for uid in ["user-alice", "user-bob"]:
                if uid in line:
                    uids_seen.add(uid)
        
        print(f"\n  UIDs observed: {sorted(uids_seen)}")
        if uids_seen == {"user-alice", "user-bob"}:
            print(f"  ✅ Both users' UIDs observed in hooks")
        else:
            print(f"  ❌ Missing UIDs — expected both user-alice and user-bob")
            all_fired = False

        print("=" * 50)
        if not all_fired:
            print("❌ SOME HOOKS DID NOT FIRE")
            sys.exit(1)
        print("✅ ALL HOOKS FIRED SUCCESSFULLY")
        
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
