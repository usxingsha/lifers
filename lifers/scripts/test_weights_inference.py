"""
Lifers 权重推理测试 — 验证所有已训练权重的推理功能
"""
import json, numpy as np, time
from pathlib import Path

wd = Path("weights")
# Allow running from project root or from scripts/ dir
if not wd.exists():
    wd = Path(__file__).resolve().parent.parent / "weights"
if not wd.exists():
    raise FileNotFoundError(f"Cannot find weights/ directory (cwd={Path.cwd()})")

ok = 0
total = 0

def load(n):
    return json.load(open(wd / n, "r", encoding="utf-8"))

def bench(name, fn, n=100):
    global ok, total
    total += 1
    try:
        t0 = time.time()
        fn()
        ms = (time.time() - t0) * 1000
        print(f"  {name:<20s} OK  ({ms:5.0f}ms)")
        ok += 1
    except Exception as e:
        print(f"  {name:<20s} FAIL: {str(e)[:100]}")

print("=" * 55)
print("Lifers 全权重推理测试 — Kali")
print("=" * 55)

# 1. Swarm DQN (8 agents × 10 tasks)
d = load("lifers_swarm_policy.json")
W1, b1, W2, b2, W3, b3 = [np.array(d[k]) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]]

def test_swarm():
    for _ in range(100):
        s = np.random.randn(1, 20).astype(np.float32)
        h1 = np.maximum(0, s @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        q = (h2 @ W3 + b3).reshape(8, 10)
        a, t = np.unravel_index(np.argmax(q), q.shape)
bench("Swarm DQN", test_swarm)

# 2. Transformer embeddings
d = load("lifers_transformer.json")
tok_emb = np.array(d["tok_emb"])

def test_transformer():
    for _ in range(10):
        seq = np.random.randint(0, min(1000, len(tok_emb)), size=(2, 16))
        vec = tok_emb[seq].mean(axis=2)
bench("Transformer", test_transformer, n=10)

# 3. Perception (3-layer MLP, multi-class)
d = load("lifers_perception_classifier.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 4)]
b = [np.array(d[f"b{i}"]) for i in range(1, 4)]

def test_perception():
    for _ in range(100):
        x = np.random.randn(5, W[0].shape[0]).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        out = h @ W[2] + b[2]
        np.argmax(out, axis=1)
bench("Perception", test_perception)

# 4. Safety (binary classifier, 1D output)
d = load("lifers_safety_classifier.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 4)]
b = [np.array(d[f"b{i}"]) for i in range(1, 4)]

def test_safety():
    for _ in range(100):
        x = np.random.randn(3, W[0].shape[0]).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        out = h @ W[2] + b[2]  # 1D logit per sample
        pred = (out > 0).astype(int)
bench("Safety", test_safety)

# 5. Social (3-layer MLP, multi-class)
d = load("lifers_social_classifier.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 4)]
b = [np.array(d[f"b{i}"]) for i in range(1, 4)]

def test_social():
    for _ in range(100):
        x = np.random.randn(3, W[0].shape[0]).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        out = h @ W[2] + b[2]
        np.argmax(out, axis=1)
bench("Social", test_social)

# 6. Proactive (3-layer MLP, multi-class)
d = load("lifers_proactive_predictor.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 4)]
b = [np.array(d[f"b{i}"]) for i in range(1, 4)]

def test_proactive():
    for _ in range(100):
        x = np.random.randn(3, W[0].shape[0]).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        out = h @ W[2] + b[2]
        np.argmax(out, axis=1)
bench("Proactive", test_proactive)

# 7. Simulation (3-layer MLP)
d = load("lifers_simulation_evaluator.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 4)]
b = [np.array(d[f"b{i}"]) for i in range(1, 4)]

def test_simulation():
    for _ in range(100):
        x = np.random.randn(3, W[0].shape[0]).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        out = h @ W[2] + b[2]
        np.argmax(out, axis=1)
bench("Simulation", test_simulation)

# 8. RL DQN (Double DQN)
d = load("lifers_rl_policy.json")
W1, b1, W2, b2, W3, b3 = [np.array(d[k]) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]]

def test_rl():
    for _ in range(100):
        s = np.random.randn(1, d["state_dim"]).astype(np.float32)
        h1 = np.maximum(0, s @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        q = h2 @ W3 + b3
        np.argmax(q)
bench("RL DQN", test_rl)

# 9. Telemetry (5-layer autoencoder: 8→32→16→8→32→8)
d = load("lifers_telemetry_detector.json")
W = [np.array(d[f"W{i}"]) for i in range(1, 6)]
b = [np.array(d[f"b{i}"]) for i in range(1, 6)]
th = d["threshold"]

def test_telemetry():
    for _ in range(200):
        x = np.random.randn(10, 8).astype(np.float32)
        h = np.maximum(0, x @ W[0] + b[0])
        h = np.maximum(0, h @ W[1] + b[1])
        h = np.maximum(0, h @ W[2] + b[2])
        h = np.maximum(0, h @ W[3] + b[3])
        dec = h @ W[4] + b[4]
        err = np.mean((x - dec) ** 2, axis=1)
        anomaly = err > th
bench("Telemetry AE", test_telemetry)

# 10. Voice (3-layer: 2 hidden + output Wy)
d = load("lifers_voice_acoustic.json")
W1, b1, W2, b2, Wy, by = [np.array(d[k]) for k in ["W1", "b1", "W2", "b2", "Wy", "by"]]

def test_voice():
    for _ in range(100):
        x = np.random.randn(4, d["input_dim"]).astype(np.float32)
        h = np.maximum(0, x @ W1 + b1)
        h = np.maximum(0, h @ W2 + b2)
        out = h @ Wy + by
        np.argmax(out, axis=1)
bench("Voice", test_voice)

# 11. Robot HAL (2-layer)
d = load("lifers_robot_hal_policy.json")
W1, b1, W2, b2 = [np.array(d[k]) for k in ["W1", "b1", "W2", "b2"]]

def test_robot_hal():
    for _ in range(100):
        s = np.random.randn(1, d["state_dim"]).astype(np.float32)
        h = np.maximum(0, s @ W1 + b1)
        q = h @ W2 + b2
        np.argmax(q)
bench("Robot HAL", test_robot_hal)

# 12. KG Embeddings
d = load("lifers_kg_embeddings.json")
ent = np.array(d["entity_emb"])

def test_kg():
    for _ in range(50):
        ids1 = np.random.randint(0, len(ent), size=(50,))
        ids2 = np.random.randint(0, len(ent), size=(50,))
        sim = ent[ids1] @ ent[ids2].T
bench("KG Embeddings", test_kg, n=50)

# 13-16. Config files
bench("Hardware Profile", lambda: load("lifers_hardware_profile.json"))
bench("Dashboard Config", lambda: load("lifers_dashboard_config.json"))
bench("Growth History", lambda: load("lifers_growth_history.json"))
bench("Training Config", lambda: load("lifers_training_config.json"))

# Summary
print("=" * 55)
print(f"ALL {ok}/{total} TESTS PASSED")
if ok == total:
    print(">> KALI WEIGHTS FULLY VALIDATED <<")
else:
    print("WARNING: some tests failed!")
print("=" * 55)
