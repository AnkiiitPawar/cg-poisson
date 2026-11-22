"""Plot the output of the `convergence` app.

usage: python scripts/plot.py [data_dir] [output_png]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/data")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/convergence.png")


def rows(name):
    with open(data / name) as fh:
        return list(csv.DictReader(fh))


history = defaultdict(list)
for r in rows("history.csv"):
    history[r["solver"]].append(float(r["rel_residual"]))
iters = defaultdict(list)
for r in rows("iterations.csv"):
    iters[r["solver"]].append((int(r["n"]), int(r["iterations"]), float(r["ms"])))
order = np.array([(float(r["h"]), float(r["max_error"])) for r in rows("order.csv")])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
colour = {name: f"C{k}" for k, name in enumerate(iters)}  # same colour per solver in every panel
for name, h in history.items():
    ax[0].semilogy(np.arange(1, len(h) + 1), h, label=name, color=colour[name])
ax[0].set(xlabel="iteration", ylabel="relative residual", title="Convergence on a 255 × 255 grid",
          xscale="log")
ax[0].legend()

for name, pts in iters.items():
    n, it, _ = zip(*pts)
    ax[1].loglog(n, it, "o-", label=name, color=colour[name])
n_ref = np.array([31, 511])
ax[1].loglog(n_ref, 4 * n_ref**2, "k:", lw=0.8, label=r"$\propto n^2$")
ax[1].loglog(n_ref, 2 * n_ref, "k--", lw=0.8, label=r"$\propto n$")
ax[1].set(xlabel="grid points per side n", ylabel="iterations to 1e-8", title="Cost to converge vs grid size")
ax[1].set_xticks([31, 63, 127, 255, 511], ["31", "63", "127", "255", "511"])
ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].legend(fontsize=8)

ax[2].loglog(order[:, 0], order[:, 1], "o-", label="max error")
ax[2].loglog(order[:, 0], order[0, 1] * (order[:, 0] / order[0, 0]) ** 2, "k--", lw=0.8, label=r"$\propto h^2$")
ax[2].set(xlabel="grid spacing h", ylabel=r"$\max|u_h - u|$", title="Second-order accuracy")
ax[2].legend()

fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}")
