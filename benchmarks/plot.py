import matplotlib.pyplot as plt

n_base = [1, 2, 5, 10]
agg_base = [35.7, 38.7, 42.6, 42.6]

n_batch = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]
agg_batch = [37.6, 73.8, 124.0, 132.4, 154.8, 196.6, 229.8, 267.8, 203.8, 238.1, 259.2]

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

ax.plot(n_batch, agg_batch, marker="o", lw=2.5, ms=6,
        color="#2563eb", label="Continuous batching")
ax.plot(n_base, agg_base, marker="o", lw=2.5, ms=6,
        color="#9ca3af", label="Baseline (one loop per request)")

# peak
ax.annotate("268 tok/s\n6.3× baseline",
            xy=(30, 267.8), xytext=(33, 285),
            fontsize=11, color="#2563eb", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5))

# memory wall
ax.axvspan(30, 60, color="#ef4444", alpha=0.06)
ax.text(45, 110, "KV cache memory wall\n(10.7 GB resident)",
        ha="center", fontsize=10, color="#b91c1c")

ax.set_xlabel("Concurrent requests", fontsize=12)
ax.set_ylabel("Aggregate throughput (tokens/sec)", fontsize=12)
ax.set_title("Continuous batching on Qwen3-4B (MLX, Mac Mini M4)",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=11, frameon=False, loc="upper left")
ax.grid(alpha=0.25, lw=0.6)
ax.set_ylim(0, 310)
ax.set_xlim(0, 63)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("throughput.png", bbox_inches="tight")