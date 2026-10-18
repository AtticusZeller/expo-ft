"""Precompute the RoboTwin seeds whose scripted expert succeeds, with each seed's instruction.

RoboTwinEnv with ROBOTWIN_SEED_TABLE then resets from this table instead of running the expert
(~20-40 s) at every reset, as RLinf does with its train_seeds.json. Seeds are checked exactly as
RoboTwinEnv.reset would check them, so the table lists the same seeds in the same order.

Runs in RoboTwin's Python environment, one process per GPU, each on a contiguous seed block:

    python scripts/robotwin/make_seed_table.py check --task beat_block_hammer --start 10100000 \
        --count 75 --out shard_0.json
    python scripts/robotwin/make_seed_table.py merge --out table.json shard_*.json
"""

import argparse
import json
import os
import sys
import time


def check(args):
    from client.envs.robotwin_env import RoboTwinEnv

    root = os.path.abspath(os.environ.get("ROBOTWIN_ROOT", "../robotwin"))
    out = os.path.abspath(args.out)
    env = RoboTwinEnv(root, args.task, task_config_name=args.task_config, instruction_type=args.instruction_type)
    seeds = []
    for seed in range(args.start, args.start + args.count):
        t0 = time.time()
        ok, episode_info = env._expert_succeeds(seed)
        if ok:
            seeds.append([seed, env._instruction(episode_info, seed)])
        print(f"seed {seed} ok={ok} {time.time() - t0:.1f}s", flush=True)
    table = {"task": args.task, "task_config": args.task_config, "instruction_type": args.instruction_type,
             "checked": [args.start, args.start + args.count], "seeds": seeds}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    print(f"{len(seeds)}/{args.count} seeds kept -> {out}")


def merge(args):
    parts = []
    for path in args.shards:
        with open(path, encoding="utf-8") as f:
            parts.append(json.load(f))
    parts.sort(key=lambda p: p["checked"][0])
    keys = {(p["task"], p["task_config"], p["instruction_type"]) for p in parts}
    if len(keys) != 1:
        sys.exit(f"Shards disagree on task/config/instruction type: {keys}")
    for a, b in zip(parts, parts[1:]):
        if a["checked"][1] != b["checked"][0]:
            sys.exit(f"Seed blocks are not contiguous: {a['checked']} then {b['checked']}")
    task, task_config, instruction_type = keys.pop()
    table = {"task": task, "task_config": task_config, "instruction_type": instruction_type,
             "checked": [parts[0]["checked"][0], parts[-1]["checked"][1]],
             "seeds": [e for p in parts for e in p["seeds"]]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    n = table["checked"][1] - table["checked"][0]
    print(f"{len(table['seeds'])}/{n} seeds in {table['checked']} -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="Check one contiguous seed block")
    c.add_argument("--task", required=True)
    c.add_argument("--task_config", default="demo_clean")
    c.add_argument("--instruction_type", default="seen")
    c.add_argument("--start", type=int, required=True)
    c.add_argument("--count", type=int, required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("merge", help="Merge contiguous shards into one table")
    m.add_argument("--out", required=True)
    m.add_argument("shards", nargs="+")
    args = parser.parse_args()
    check(args) if args.cmd == "check" else merge(args)


if __name__ == "__main__":
    main()
