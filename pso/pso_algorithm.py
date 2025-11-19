import numpy as np
from .utils import PSOUtils
from .particle import Particle

class PSO:
    def __init__(self, w, c2, r2, rand_check, posisi_awal):
        self.w = w
        self.c2 = c2
        self.r2 = r2
        self.rand_check = rand_check

        # daftar partikel
        self.particles = [Particle(pos) for pos in posisi_awal]

        # gBest awal
        self.update_gbest()

    # ------------------------------------------------------------------
    def update_gbest(self):
        """Menentukan global best berdasarkan pBest semua partikel."""
        pbest_vals = [p.pbest_val for p in self.particles]
        idx = np.argmax(pbest_vals)

        self.gbest = self.particles[idx].pbest.copy()
        self.gbest_val = pbest_vals[idx]

    # ------------------------------------------------------------------
    def iterasi(self, it):
        PSOUtils.print_header(f"ITERASI {it}")

        # ------------------ UPDATE KECEPATAN ------------------
        print("\nUpdate Kecepatan:")
        print("-" * 65)

        for i, p in enumerate(self.particles):
            p.update_kecepatan(self.w, self.c2, self.r2, self.gbest)
            print(f"V{i+1}: {p.kecepatan}")

        # ------------------ UPDATE POSISI ---------------------
        print("\nSigmoid(V) dan Update Posisi:")
        print("-" * 65)

        posisi_baru = []
        fitness_baru = []

        for i, p in enumerate(self.particles):
            sigmoid_val, pos = p.update_posisi(self.rand_check)
            posisi_baru.append(pos)
            fitness_baru.append(PSOUtils.fitness(PSOUtils.decode(pos)))
            print(f"P{i+1} sigmoid: {np.round(sigmoid_val,4)} → posisi baru: {pos}")

        PSOUtils.print_table("\nPosisi Baru + Fitness:", posisi_baru, fitness_baru)

        # ------------------ UPDATE PBEST ----------------------
        print("\nUpdate pBest:")
        print("-" * 65)

        for i, p in enumerate(self.particles):
            before = p.pbest_val
            p.cek_pbest()
            if p.pbest_val > before:
                print(f"P{i+1}: pBest diperbarui menjadi {p.pbest}")
            else:
                print(f"P{i+1}: tetap (tidak lebih baik dari pBest sebelumnya)")

        # ------------------ UPDATE GBEST ----------------------
        self.update_gbest()

        print("\nStatus pBest Setelah Update:")
        PSOUtils.print_table(
            "pBest:",
            [p.pbest for p in self.particles],
            [p.pbest_val for p in self.particles]
        )

        print(f"gBest baru: {self.gbest}   fitness={self.gbest_val}")
