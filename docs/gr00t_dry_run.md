# GR00T cloud dry run (before the real data)

**Goal:** run the whole GR00T cloud pipeline once on a small **fake dataset**, so any
setup problems get fixed now instead of on the real run. The arm isn't needed.

The fake dataset is `software/gr00t/dryrun/pick_ball_sim`: 20 simulated demos (11 ball,
9 block) made with `autodemo.py --sim`. Its camera frames are just a moving orange square,
so the model **learns nothing useful** from it. We're only checking that every step
runs: install, download, fine-tune, save a checkpoint, serve, and the laptop talking to it.

**Time:** about 1–1.5 hours, mostly waiting. **Cost:** about $1–4 of GPU time, or free credits.

---

## Before you start
- [ ] Hugging Face **Read** token: huggingface.co → Settings → Access Tokens → New
      token → type "Read". Starts with `hf_`. Don't paste it anywhere but the cloud terminal.
- [ ] A Linux GPU machine with **≥ 40 GB GPU memory** (L40S, A100 or H100) and **≥ 100 GB
      disk**, that you can SSH into (Lightning AI Studio, RunPod, Lambda, …).
      Note its SSH command, user, host and port.
- [ ] On the laptop, get the latest code (it includes the fake dataset):
      ```
      cd gr00t-arm
      git pull
      ```

## 1. Copy the GR00T folder up (laptop, PowerShell)
```
cd gr00t-arm
scp -P <port> -r software\gr00t <user>@<host>:~/
```
This copies the scripts **and** the fake dataset (in `gr00t/dryrun/`).

## 2. Set up the machine (cloud terminal, ~20 min)
```
ssh -p <port> <user>@<host>
export HF_TOKEN=hf_your_token
bash ~/gr00t/setup_cloud.sh
```
✅ **Expect:** `GR00T installed`, then (after the model downloads) `setup done`.

## 3. Short fine-tune (cloud, ~10–20 min)
Use `tmux` so it keeps running if your connection drops (`tmux attach` to get back).
```
tmux
MAX_STEPS=200 SAVE_STEPS=100 bash ~/gr00t/finetune.sh ~/gr00t/dryrun/pick_ball_sim
```
✅ **Expect:** training steps counting up to 200, then `done: newest checkpoint
.../gr00t_arm_ckpt/checkpoint-200`. The loss numbers don't matter for the dry run.
The eval plot at the end may fail on fake data; that's OK.

## 4. Serve the model (cloud)
```
bash ~/gr00t/serve.sh
```
✅ **Expect:** `serving .../checkpoint-200 on port 5555`, and it keeps running. Leave it.

## 5. Connect from the laptop (two PowerShell windows)
**Window 1, the tunnel.** It just stays open, with no output:
```
ssh -p <port> -N -L 5555:localhost:5555 <user>@<host>
```
**Window 2, the test.** Use `--sim`, so no arm or camera is used:
```
cd gr00t-arm\software
uv run run_policy.py gr00t --host localhost --sim --task "pick up the ball" --seconds 5
```
✅ **Expect:** `connected to GR00T at localhost:5555`, then a few lines like
`policy call 150 ms, 8 steps, grip 120`. That's the whole pipeline working.

⚠️ Keep `--sim` on. This fake-trained model would move a real arm randomly. The safety
check blocks dangerous poses, but there's no reason to risk it.

## 6. Tell Claude
Send:
- the **policy call ms** numbers (how fast it answers),
- any **error text**, copied in full, from any step,
- which provider and GPU you used.

## 7. Stop the machine
Stop or delete it in the provider's dashboard. **It bills until you do.**
If you **stop** it but keep the disk, the real run can skip step 2, saving ~20 min
(a stopped disk costs cents per day). Delete the dry-run checkpoint first
(`rm -rf ~/gr00t_arm_ckpt`), so `serve.sh` doesn't pick it up later.

---

## If something goes wrong
| Problem | Fix |
|---|---|
| `GatedRepoError` / 401 / 403 during setup | Token wrong or not "Read", or access to `nvidia/Cosmos-Reason2-2B` not granted. |
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"`, then rerun the command. |
| `CUDA out of memory` while training | Add `BATCH=16` in front: `BATCH=16 MAX_STEPS=200 SAVE_STEPS=100 bash ...` |
| `no checkpoint in .../gr00t_arm_ckpt` | `SAVE_STEPS` was bigger than `MAX_STEPS`; use the numbers above. |
| Disk full | The machine needs ≥ 100 GB; pick a bigger disk. |
| `GR00T server not reachable` on the laptop | Is `serve.sh` still running? Is the tunnel window still open? Right port? |
| `ssh` asks for a password / is refused | Use the exact SSH command the provider shows (user, host, port, key). |
| Anything else | Copy the full error to Claude. |
