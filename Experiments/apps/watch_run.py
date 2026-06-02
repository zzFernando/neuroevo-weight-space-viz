#!/usr/bin/env python3
"""Watch evosax run progress from log file."""
import re
import sys
import time
from pathlib import Path

LOG = Path("/tmp/evosax_big.log")
GENS = 500
SEEDS = 3

RE_GEN   = re.compile(r"gen\s+(\d+)/(\d+)\s+best=\s*([\d.+-]+)\s+mean=\s*([\d.+-]+)\s+\(([\d.]+)s\)")
RE_SAVED = re.compile(r"saved →\s+(.+\.npz)")
RE_ALGO  = re.compile(r"=== (.+) ===")

BAR = 30

def bar(frac):
    filled = int(BAR * frac)
    return "█" * filled + "░" * (BAR - filled)

def clear():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

RE_WARMUP = re.compile(r"JIT warmup done")

state = {"algo": "", "seed_done": 0, "gen": 0, "best": 0.0,
         "mean": 0.0, "elapsed": 0.0, "saved": [], "warming": False, "warm_done": False}

jit_start: float | None = None

print(f"Watching {LOG} … (Ctrl+C to stop)\n")

while True:
    if not LOG.exists():
        print(f"\r⏳ aguardando {LOG} …", end="", flush=True)
        time.sleep(2)
        continue

    lines = LOG.read_text().splitlines()

    for line in lines:
        m = RE_ALGO.search(line)
        if m:
            state["algo"]      = m.group(1)
            state["warm_done"] = False
            state["warming"]   = True
            state["gen"]       = 0

        if RE_WARMUP.search(line):
            state["warm_done"] = True
            state["warming"]   = False

        m = RE_GEN.search(line)
        if m:
            state["gen"]     = int(m.group(1))
            state["best"]    = float(m.group(3))
            state["mean"]    = float(m.group(4))
            state["elapsed"] = float(m.group(5))

        m = RE_SAVED.search(line)
        if m:
            name = Path(m.group(1).strip()).name
            if name not in state["saved"]:
                state["saved"].append(name)
                state["seed_done"] = len(state["saved"])

    g   = state["gen"]
    el  = state["elapsed"]
    frac = g / GENS if GENS > 0 else 0
    eta  = (el / g * (GENS - g)) if g > 0 else 0

    # JIT warmup spinner
    jit_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    spinner = jit_frames[int(time.time() * 3) % len(jit_frames)]

    clear()
    print("━" * 52)
    print(f"  open_es  pop=128  gens={GENS}  seeds={SEEDS}")
    print("━" * 52)
    print(f"\n  Seed atual : {state['algo'] or '…'}")

    if state["warming"] and not state["warm_done"]:
        print(f"\n  {spinner} JIT warmup … (compilando rollout vmapado)")
        print(f"  Geração    : aguardando warmup")
        print(f"  Best fit   : —       Mean: —")
        print(f"  Tempo      : —       ETA:  —")
    else:
        status = "✓ JIT pronto" if state["warm_done"] or g > 0 else "…"
        print(f"  JIT warmup : {status}")
        print(f"  Geração    : {g}/{GENS}  [{bar(frac)}]  {frac*100:.1f}%")
        print(f"  Best fit   : {state['best']:+.2f}    Mean: {state['mean']:+.2f}")
        print(f"  Tempo      : {el/60:.1f} min  |  ETA: ~{eta/60:.1f} min")
    print()
    print(f"  Seeds concluídos: {state['seed_done']}/{SEEDS}")
    for s in state["saved"]:
        print(f"    ✓ {s}")
    remaining = SEEDS - state["seed_done"]
    if remaining > 0 and el > 0 and g > 0:
        per_run = el / 60
        print(f"\n  Estimativa restante: ~{remaining * per_run:.0f} min")
    if state["seed_done"] == SEEDS:
        print("\n  ✅ TODOS OS SEEDS CONCLUÍDOS")
        break
    print("\n" + "━" * 52)

    time.sleep(5)
