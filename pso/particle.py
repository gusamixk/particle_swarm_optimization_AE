import numpy as np
from .utils import PSOUtils

class Particle:
    def __init__(self, posisi_awal):
        self.posisi = np.array(posisi_awal, dtype=float)
        self.kecepatan = np.zeros_like(self.posisi)

        # pBest awal
        self.pbest = self.posisi.copy()
        self.pbest_val = PSOUtils.fitness(PSOUtils.decode(self.pbest))

    # ----------------------------------------------------------------------
    def update_kecepatan(self, w, c1, r1, c2, r2, gbest):
        """Update velocity sesuai versi dosen (x = pbest tiap iterasi)."""
        x = self.pbest  # ❗ DOSEN VERSION: x = pBest, bukan posisi real

        cognitive = c1 * r1 * (self.pbest - x)       # = 0
        social = c2 * r2 * (gbest - x)

        self.kecepatan = w * self.kecepatan + cognitive + social

    # ----------------------------------------------------------------------
    def update_posisi(self, rand_check):
        """Update posisi dengan sigmoid + ambang Rand, lalu posisi langsung = pBest."""
        s = PSOUtils.sigmoid(self.kecepatan)
        new_posisi = PSOUtils.update_posisi(s, rand_check)

        # ❗ INI ATURAN DOSEN:
        # Setelah dapat posisi baru, nilai X langsung dianggap sebagai pBest
        self.posisi = new_posisi.copy()

        return s, self.posisi

    # ----------------------------------------------------------------------
    def cek_pbest(self):
        """Update pBest jika posisi baru lebih baik."""
        nilai_baru = PSOUtils.fitness(PSOUtils.decode(self.posisi))
        if nilai_baru > self.pbest_val:
            self.pbest = self.posisi.copy()
            self.pbest_val = nilai_baru
