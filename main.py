from pso.pso_algorithm import PSO
from pso.utils import PSOUtils
import numpy as np

# ============================
# Parameter PSO
# ============================
w = 0.5
c2 = 1.0

r2 = np.array([0.9, 0.7, 0.5, 0.3])
rand_check = np.array([0.7, 0.2, 0.8, 0.4])

posisi_awal = [
    [0, 0, 1, 0],   # 2
    [1, 0, 0, 0],   # 8
    [1, 1, 0, 0]    # 12
]

# ============================
# Inisialisasi PSO
# ============================
pso = PSO(w, c2, r2, rand_check, posisi_awal)

# Menampilkan Inisialisasi (Iterasi 0)
PSOUtils.print_header("ITERASI 0 (Inisialisasi)")
PSOUtils.print_table(
    "pBest Awal:",
    [p.pbest for p in pso.particles],
    [p.pbest_val for p in pso.particles]
)
print(f"gBest: {pso.gbest}   fitness={pso.gbest_val}")

# Jalankan iterasi
pso.iterasi(1)
pso.iterasi(2)

PSOUtils.print_header("PROSES SELESAI")
