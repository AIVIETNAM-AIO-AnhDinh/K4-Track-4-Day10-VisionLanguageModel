"""Số liệu bổ sung cho REPORT.md, tính lại từ work/*.jsonl và data/splits (không cần GPU).

Chạy: python report/analysis.py
"""
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import CLASSES, read_jsonl  # noqa: E402
from scripts.score import _gt_boxes, _match, _prf, evaluate, parse_output  # noqa: E402

DATA, WORK = ROOT / "data", ROOT / "work"
splits = {s: read_jsonl(DATA / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
TEST = splits["test"]


def preds(key):
    return {r["image_id"]: r["raw"] for r in read_jsonl(WORK / f"{key}.jsonl")}


def f1(res):
    return round(res["micro"]["f1"], 3)


print("== Câu 3: lệch thang toạ độ ==")
for key in ("e0", "e3"):
    p = preds(key)
    print(f"{key}: chấm norm1000 F1={f1(evaluate(TEST, p, 'norm1000', n_boot=1))} | "
          f"chấm như pixel F1={f1(evaluate(TEST, p, 'pixel', n_boot=1))}")


def oracle(coord):
    """Đáp án đúng (nhãn GT) viết theo `coord`."""
    out = {}
    for e in TEST:
        sx, sy = (1000, 1000) if coord == "norm1000" else (e["width"], e["height"])
        out[e["image_id"]] = json.dumps([{"label": o["label"], "bbox_2d": [
            round((o["yolo"][0] - o["yolo"][2] / 2) * sx), round((o["yolo"][1] - o["yolo"][3] / 2) * sy),
            round((o["yolo"][0] + o["yolo"][2] / 2) * sx), round((o["yolo"][1] + o["yolo"][3] / 2) * sy)]}
            for o in e["objects"]])
    return out


for write, score in (("norm1000", "norm1000"), ("norm1000", "pixel"), ("pixel", "norm1000"), ("pixel", "pixel")):
    print(f"GT viết theo {write:8s} chấm --coord {score:8s}: F1={f1(evaluate(TEST, oracle(write), score, n_boot=1))}")

print("\n== Câu 6: rò rỉ nếu chia ngẫu nhiên ==")
by_sess = Counter(r["session"] for s in splits.values() for r in s)
print(f"{len(by_sess)} phiên, ảnh/phiên: min {min(by_sess.values())} trung vị {statistics.median(by_sess.values())} "
      f"max {max(by_sess.values())}")
allrows = [r for s in splits.values() for r in s]
seq = defaultdict(list)
for r in allrows:
    seq[r["session"]].append(r["image_id"])
for s in seq:
    seq[s].sort()
neighbor = {}
for s, ids in seq.items():
    for i, x in enumerate(ids):
        neighbor[x] = {ids[j] for j in (i - 1, i + 1) if 0 <= j < len(ids)}
leaks = []
for seed in range(20):
    ids = [r["image_id"] for r in allrows]
    random.Random(seed).shuffle(ids)
    n_tr = len(splits["train"])
    tr, te = set(ids[:n_tr]), ids[-len(TEST):]
    leaks.append(sum(bool(neighbor[x] & tr) for x in te) / len(te))
print(f"chia ngẫu nhiên (cùng kích thước): {statistics.mean(leaks):.1%} ảnh test có frame liền kề nằm trong train "
      f"(20 seed, {min(leaks):.1%}–{max(leaks):.1%}); chia theo phiên: 0%")
try:
    import numpy as np
    from PIL import Image

    def small(i):
        e = next(r for r in allrows if r["image_id"] == i)
        return np.asarray(Image.open(DATA / e["image_path"]).convert("L").resize((64, 36)), dtype=np.float32)

    rng = random.Random(0)
    adj, rnd = [], []
    for _ in range(200):
        s = rng.choice([k for k, v in seq.items() if len(v) > 2])
        i = rng.randrange(len(seq[s]) - 1)
        adj.append(np.abs(small(seq[s][i]) - small(seq[s][i + 1])).mean())
        a, b = rng.sample(allrows, 2)
        rnd.append(np.abs(small(a["image_id"]) - small(b["image_id"])).mean())
    print(f"sai khác điểm ảnh TB (thang 0–255, ảnh thu nhỏ 64×36): frame liền kề {np.median(adj):.1f} | "
          f"hai ảnh ngẫu nhiên {np.median(rnd):.1f}")
except ImportError:
    pass

print("\n== Câu 8: bootstrap ghép cặp (cùng mẫu ảnh cho hai setup) ==")


def rows(key):
    p = preds(key)
    out = []
    for e in TEST:
        per, _ = _match(_gt_boxes(e), parse_output(p.get(e["image_id"])) or [])
        out.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return out


R = {k: rows(k) for k in ("e0", "e1", "e2", "e3", "e4_data_size_0.1", "e5")}


def micro(rs, idx):
    return _prf(*(sum(rs[i][k] for i in idx) for k in range(3)))["f1"]


def paired(a, b, n_boot=2000, seed=0):
    rng, n = random.Random(seed), len(TEST)
    d = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(micro(R[b], idx) - micro(R[a], idx))
    d.sort()
    full = micro(R[b], range(n)) - micro(R[a], range(n))
    lo, hi = d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]
    print(f"{b:17s} − {a:17s}: ΔF1 = {full:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  "
          f"P(Δ≤0) = {sum(x <= 0 for x in d) / n_boot:.3f}")


for a, b in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3"), ("e4_data_size_0.1", "e3"), ("e3", "e5")):
    paired(a, b)
for k in ("e0", "e3"):
    ev = json.loads((ROOT / "report" / "evidence" / f"{k}.json").read_text())
    print(f"{k}: CI95 rộng {ev['f1_ci95'][1] - ev['f1_ci95'][0]:.3f}")

print("\n== Câu 9: kích thước box theo lớp (test) và recall E3 theo kích thước ==")
area = defaultdict(list)
for e in TEST:
    for o in e["objects"]:
        area[o["label"]].append(o["yolo"][2] * o["yolo"][3] * 100)
for c in CLASSES:
    tr = sum(o["label"] == c for e in splits["train"] for o in e["objects"])
    print(f"{c:8s} train {tr:5d} box | test {len(area[c]):3d} box, diện tích trung vị {statistics.median(area[c]):.2f}% ảnh")
p3 = preds("e3")
buckets = defaultdict(lambda: [0, 0])
for e in TEST:
    pr = parse_output(p3.get(e["image_id"])) or []
    for g in _gt_boxes(e):
        a = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1]) / 1e4
        b = "<0.5%" if a < 0.5 else "0.5–2%" if a < 2 else "≥2%"
        hit = any(p["label"] == g["label"] and _match([g], [p])[0][g["label"]][0] for p in pr)
        buckets[(g["label"], b)][0] += hit
        buckets[(g["label"], b)][1] += 1
for c in CLASSES:
    print(c, " | ".join(f"{b}: {buckets[(c, b)][0]}/{buckets[(c, b)][1]}" for b in ("<0.5%", "0.5–2%", "≥2%")))
conf = Counter()
for e in TEST:
    pr = parse_output(p3.get(e["image_id"])) or []
    for p in pr:
        best = max(((_match([g], [dict(p, label=g["label"])])[0][g["label"]][0], g["label"]) for g in _gt_boxes(e)),
                   default=(0, None))
        if best[0] and best[1] != p["label"]:
            conf[(p["label"], best[1])] += 1
print("E3 box đúng vị trí (IoU≥0,5) nhưng sai lớp (dự đoán→thật):", dict(conf))

print("\n== E5: reward GRPO ==")
run = json.loads((WORK / "runs" / "e5" / "run.json").read_text())
logs = [h for h in run["log_history"] if "reward" in h]
print("reward theo bước:", [(h["step"], round(h["reward"], 3)) for h in logs])
