import numpy as np

class PSOUtils:
    @staticmethod
    def decode(bits):
        """Mengubah representasi biner 4-bit menjadi bilangan desimal."""
        bobot = [8, 4, 2, 1]
        return int(sum(int(b) * w for b, w in zip(bits, bobot)))

    @staticmethod
    def fitness(x):
        """Fungsi objektif: f(x) = (x - 5)^2 + 10."""
        return (x - 5)**2 + 10

    @staticmethod
    def sigmoid(v):
        """Fungsi sigmoid untuk probabilitas update posisi."""
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def update_posisi(sigmoid_val, rand_check):
        """Menentukan posisi baru berdasarkan probabilitas sigmoid."""
        result = np.zeros_like(sigmoid_val)
        for i in range(len(sigmoid_val)):
            result[i] = 1 if rand_check[i] < sigmoid_val[i] else 0
        return result

    # ----------------------------------------------------------------------
    # Fungsi Perapihan Output
    # ----------------------------------------------------------------------
    @staticmethod
    def print_header(title):
        print("\n" + "=" * 65)
        print(title)
        print("=" * 65)

    @staticmethod
    def print_table(label, posisi, fit):
        print(f"\n{label}")
        print("-" * 65)
        for i in range(len(posisi)):
            print(
                f"P{i+1}  Posisi: {posisi[i]}   "
                f"x={PSOUtils.decode(posisi[i]):2d}   f(x)={fit[i]:3d}"
            )
        print("-" * 65)
