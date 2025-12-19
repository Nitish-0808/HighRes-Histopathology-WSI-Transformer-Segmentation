import csv
from pathlib import Path

class AvgMeter:
    def __init__(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += float(val) * k; self.n += k
    @property
    def avg(self): return (self.sum / max(1, self.n))
