# VM Workflow Guide — fairinf-sglang-upstream

Quick reference for pushing changes, syncing to the GCP VM, and running tests.

---

## GCP VM Details

- **Name:** `sglang-hooks-spot`
- **Zone:** `europe-west4-a`
- **Project:** `delta-fair-inference`
- **Type:** a2-highgpu-1g (1x A100 40GB)
- **Lifecycle:** Spot instance with auto-shutdown after 30min idle or 4h max runtime

---

## Workflow: Local → VM → Test

### 1. **Make changes locally**

Work in `/home/dev/project/verification/fairinf-sglang-upstream` on the `verity/minimal-scheduling-hooks` branch.

### 2. **Commit and push to GitHub**

```bash
cd /home/dev/project/verification/fairinf-sglang-upstream
git add -A
git commit -m "Your commit message"
git push github verity/minimal-scheduling-hooks
```

### 3. **Start the VM** (if stopped)

```bash
gcloud compute instances start sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference
```

Wait ~20-30 seconds for SSH to be ready.

### 4. **Sync changes to VM via git bundle** (fastest method)

```bash
# Create bundle locally
cd /home/dev/project/verification/fairinf-sglang-upstream
git bundle create /tmp/sglang-hooks.bundle HEAD

# Copy to VM
gcloud compute scp /tmp/sglang-hooks.bundle sglang-hooks-spot:~/ \
  --zone=europe-west4-a \
  --project=delta-fair-inference

# Fetch and merge on VM
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="cd ~/fairinf-sglang-upstream && git fetch ~/sglang-hooks.bundle && git merge FETCH_HEAD"
```

**Alternative (if VM has GitHub access):**
```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="cd ~/fairinf-sglang-upstream && git pull"
```

### 5. **Run tests in the venv**

```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="cd ~/fairinf-sglang-upstream && source ~/sglang_venv/bin/activate && python3 test_hooks.py"
```

Or run the simple client test (assumes server is already running):
```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="cd ~/fairinf-sglang-upstream && source ~/sglang_venv/bin/activate && python3 test_client.py"
```

---

## Running the Server Manually

### Start server in background:
```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="bash ~/start_server.sh"
```

This starts the server with:
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Port: `30000`
- Logging policy: `sglang.srt.scheduling_hooks.logging_policy.LoggingSchedulingPolicy`
- Logs: `~/sglang_server.log`

### Check server logs:
```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="tail -f ~/sglang_server.log"
```

### Stop server:
```bash
gcloud compute ssh sglang-hooks-spot \
  --zone=europe-west4-a \
  --project=delta-fair-inference \
  --command="pkill -f 'sglang.launch_server'"
```

---

## First-Time VM Setup

If you need to recreate the VM from scratch:

### 1. Clone the repo
```bash
git clone https://github.com/sgl-project/sglang.git fairinf-sglang-upstream
cd fairinf-sglang-upstream
git remote add devbali https://github.com/devbali/fairinf-sglang.git
git fetch devbali verity/minimal-scheduling-hooks
git checkout -b verity/minimal-scheduling-hooks devbali/verity/minimal-scheduling-hooks
```

Or use a git bundle (when GitHub access is restricted):
```bash
# On local machine, create bundle
cd /home/dev/project/verification/fairinf-sglang-upstream
git bundle create /tmp/sglang-hooks.bundle HEAD verity/minimal-scheduling-hooks

# Copy to VM
gcloud compute scp /tmp/sglang-hooks.bundle sglang-hooks-spot:~/ \
  --zone=europe-west4-a \
  --project=delta-fair-inference

# On VM, clone from bundle
git clone ~/sglang-hooks.bundle fairinf-sglang-upstream
cd fairinf-sglang-upstream
git checkout verity/minimal-scheduling-hooks
```

### 2. Install Rust (required for building sglang)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

### 3. Create venv and install sglang
```bash
python3 -m venv ~/sglang_venv
source ~/sglang_venv/bin/activate
cd ~/fairinf-sglang-upstream
pip install --upgrade pip
pip install -e 'python[all]'
```

This takes ~10-15 minutes (large PyTorch/CUDA dependencies).

---

## Tips

- **Spot VM auto-shutdown:** The VM stops itself after 30min idle or 4h max runtime. Re-start with `gcloud compute instances start`.
- **SSH sessions:** Use `--command` for one-shot commands; omit it for an interactive shell.
- **Logs:** Check `~/sglang_server.log` for hook output (`grep '\[HOOK\]'`).
- **Model cache:** Models download to `~/.cache/huggingface/hub/` and persist across restarts.

---

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| Start VM | `gcloud compute instances start sglang-hooks-spot --zone=europe-west4-a --project=delta-fair-inference` |
| SSH into VM | `gcloud compute ssh sglang-hooks-spot --zone=europe-west4-a --project=delta-fair-inference` |
| Copy file to VM | `gcloud compute scp <local-path> sglang-hooks-spot:~/ --zone=europe-west4-a --project=delta-fair-inference` |
| Activate venv | `source ~/sglang_venv/bin/activate` |
| Run test | `cd ~/fairinf-sglang-upstream && source ~/sglang_venv/bin/activate && python3 test_hooks.py` |
| Check server status | `ps aux | grep sglang.launch_server` |
| View server logs | `tail -50 ~/sglang_server.log` |
| Stop server | `pkill -f 'sglang.launch_server'` |
| Stop VM | `gcloud compute instances stop sglang-hooks-spot --zone=europe-west4-a --project=delta-fair-inference` |
