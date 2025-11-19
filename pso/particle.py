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
    def update_kecepatan(self, w, c2, r2, gbest):
        """Update velocity berdasarkan komponen global."""
        self.kecepatan = w * self.kecepatan + c2 * r2 * (gbest - self.pbest)

    # ----------------------------------------------------------------------
    def update_posisi(self, rand_check):
        """Update posisi berdasarkan probabilitas sigmoid."""
        s = PSOUtils.sigmoid(self.kecepatan)
        self.posisi = PSOUtils.update_posisi(s, rand_check)
        return s, self.posisi

    # ----------------------------------------------------------------------
    def cek_pbest(self):
        """Memperbarui pBest jika posisi baru lebih baik."""
        nilai_baru = PSOUtils.fitness(PSOUtils.decode(self.posisi))
        if nilai_baru > self.pbest_val:
            self.pbest = self.posisi.copy()
            self.pbest_val = nilai_baru
