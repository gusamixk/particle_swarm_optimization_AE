import numpy as np

class PSOUtils:

    @staticmethod
    def decode(bits):
        weights = [8, 4, 2, 1]
        return int(sum(int(b) * w for b, w in zip(bits, weights)))

    @staticmethod
    def fitness(x):
        return (x - 5)**2 + 10

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def update_posisi(sigmoid_val, rand_check):
        return (rand_check < sigmoid_val).astype(float)

    @staticmethod
    def print_header(text):
        print("=" * 60)
        print(text)
        print("=" * 60)

    @staticmethod
    def print_table(title, posisi, fitness):
        print(title)
        print("-" * 60)
        print("Posisi".ljust(20), "| Fitness")
        print("-" * 60)
        for p, f in zip(posisi, fitness):
            print(str(p).ljust(20), "|", f)
        print("-" * 60)
