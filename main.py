from pso import PSO, PSOUtils

# ============================
# Parameter PSO
# ============================
w = 0.5
c1 = 1.0
c2 = 1.0
r1 = [0.2, 0.4, 0.6, 0.8]
r2 = [0.9, 0.7, 0.5, 0.3]
rand_check = [0.7, 0.2, 0.8, 0.4]

posisi_awal = [
    [0,0,1,0],
    [1,0,0,0],
    [1,1,0,0]
]

# ============================
# Inisialisasi PSO
# ============================
p = PSO(w, c1, c2, r1, r2, rand_check, posisi_awal)

# Tampilkan ITERASI 0
PSOUtils.print_header("ITERASI 0 (Inisialisasi)")
PSOUtils.print_table(
    "pBest Awal:",
    [x.pbest for x in p.particles],
    [x.pbest_val for x in p.particles]
)
print(f"gBest: {p.gbest}   fitness={p.gbest_val}")

# ============================
# Jalankan iterasi
# ============================
p.iterasi(1)
p.iterasi(2)
