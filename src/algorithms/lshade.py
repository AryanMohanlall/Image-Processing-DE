import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from shade import SHADE

from shade import SHADE
 
class LSHADE(SHADE):
    name = "L-SHADE"
 
    def __init__(self, *args, N_init=None, N_min=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_init = N_init if N_init is not None else self.pop_size
        self.N_min  = N_min
 
    def _shrink_population(self):
        ratio  = self.fes / self.max_fes
        N_new  = max(self.N_min,
                     int(round(self.N_init
                                + (self.N_min - self.N_init) * ratio)))
        current = len(self.pop)
        if N_new < current:
            n_drop  = current - N_new
            worst   = np.argsort(self.fitness)[:n_drop]   # ascending = worst first
            keep    = np.ones(current, dtype=bool)
            keep[worst] = False
            self.pop     = self.pop[keep]
            self.fitness = self.fitness[keep]
        self.pop_size = len(self.pop)
 
        while len(self.archive) > self.pop_size:
            self.archive.pop(0)
 
    def _generation(self, rng):
        super()._generation(rng)
        self._shrink_population()
