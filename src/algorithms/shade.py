import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DEResult:
    def __init__(self, best_x, best_f, history, fes_used, algo_name="SHADE"):
        self.best_x    = best_x
        self.best_f    = best_f
        self.history   = history
        self.fes_used  = fes_used
        self.algo_name = algo_name

    @property
    def thresholds(self):
        return np.sort(np.round(self.best_x)).astype(int)

class BaseDE:
    def __init__(self, objective, K, bounds=(1, 254), pop_size=30,
                 max_fes=10000, seed=None):
        self.objective = objective
        self.K = K
        self.lo, self.hi = bounds
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.seed = seed
        self.pop = None
        self.fitness = None
        self.fes = 0
        self.history = []

    def _clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def _eval(self, x):
        self.fes += 1
        return self.objective(x)

    def _init_population(self, rng):
        self.pop = rng.uniform(self.lo, self.hi, size=(self.pop_size, self.K))
        self.fitness = np.array([self._eval(ind) for ind in self.pop])

    def run(self):
        rng = np.random.default_rng(self.seed)
        self._init_population(rng)
        self.history = [self.fitness.max()]
        while self.fes < self.max_fes:
            self._generation(rng)
            self.history.append(self.fitness.max())
        best_idx = int(np.argmax(self.fitness))
        return DEResult(self.pop[best_idx].copy(), float(self.fitness[best_idx]),
                self.history, self.fes, self.name)

    def _generation(self, rng):
        raise NotImplementedError


class SHADE(BaseDE):
    name = "SHADE"

    def __init__(self, *args, H=10, use_archive=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = H
        self.use_archive = use_archive
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.mem_idx = 0
        self.archive = []

    def _sample_F(self, rng, r):
        while True:
            f = rng.standard_cauchy() * 0.1 + self.M_F[r]
            if f > 0:
                return min(f, 1.0)

    def _sample_CR(self, rng, r):
        return float(np.clip(rng.normal(self.M_CR[r], 0.1), 0, 1))

    def _generation(self, rng):
        NP = self.pop_size
        order = np.argsort(-self.fitness)
        S_F, S_CR, weights = [], [], []

        for i in range(NP):
            if self.fes >= self.max_fes:
                break

            r = rng.integers(self.H)
            F_i = self._sample_F(rng, r)
            CR_i = self._sample_CR(rng, r)

            p_lo = min(2.0 / NP, 0.2)
            p_hi = max(2.0 / NP, 0.2)
            p = rng.uniform(p_lo, p_hi)
            n_pbest = max(1, int(p * NP))
            pbest_idx = order[rng.integers(n_pbest)]
            x_pbest = self.pop[pbest_idx]

            idxs = [j for j in range(NP) if j != i]
            r1 = rng.choice(idxs)

            if self.use_archive and self.archive:
                combined = self.pop.tolist() + self.archive
                r2_vec = combined[rng.integers(len(combined))]
            else:
                r2 = rng.choice([j for j in idxs if j != r1])
                r2_vec = self.pop[r2]

            mutant = (self.pop[i] + F_i * (x_pbest - self.pop[i]) +
                      F_i * (self.pop[r1] - r2_vec))
            mutant = self._clip(mutant)

            cross_mask = rng.random(self.K) < CR_i
            if not cross_mask.any():
                cross_mask[rng.integers(self.K)] = True
            trial = np.where(cross_mask, mutant, self.pop[i])

            f_trial = self._eval(trial)
            if f_trial >= self.fitness[i]:
                if self.use_archive:
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > NP:
                        self.archive.pop(rng.integers(len(self.archive)))
                delta = f_trial - self.fitness[i]
                self.pop[i] = trial
                self.fitness[i] = f_trial
                S_F.append(F_i)
                S_CR.append(CR_i)
                weights.append(delta)

        if S_F:
            w = np.array(weights)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            S_F, S_CR = np.array(S_F), np.array(S_CR)
            mean_L_F = np.sum(w * S_F**2) / np.sum(w * S_F)
            mean_L_CR = (np.sum(w * S_CR**2) / np.sum(w * S_CR)
                         if np.sum(w * S_CR) > 0 else np.mean(S_CR))
            self.M_F[self.mem_idx] = mean_L_F
            self.M_CR[self.mem_idx] = mean_L_CR
            self.mem_idx = (self.mem_idx + 1) % self.H
