import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


class NeuroEvoMoons:
    """
    Neuroevolução de uma MLP pequena para classificar o dataset make_moons.
    Isso gera um espaço de pesos bem mais rico para visualização.
    """

    def __init__(
        self,
        pop_size=500,
        input_dim=2,
        hidden_dim=64,
        output_dim=1,
        mutation_rate=0.05,
        seed=42,
    ):
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # Arquitetura da rede
        self.shapes = [
            (input_dim, hidden_dim),
            (hidden_dim, output_dim),
        ]

        # Dataset não-linear (mais complexo que XOR)
        X, y = make_moons(n_samples=400, noise=0.25, random_state=seed)
        self.scaler = StandardScaler().fit(X)
        self.X = self.scaler.transform(X)
        self.y = y.reshape(-1, 1)

        # População inicial
        self.population = [self.random_individual() for _ in range(pop_size)]

    # ---------- Representação ----------

    def random_individual(self):
        # pesos iniciais ~ N(0, 0.5)
        return [self.rng.normal(0, 0.5, size=s) for s in self.shapes]

    def flatten(self, individual):
        return np.concatenate([w.ravel() for w in individual])

    # ---------- Forward / Fitness ----------

    def forward(self, individual, X):
        w1, w2 = individual
        h = np.tanh(X @ w1)
        o = h @ w2
        # sigmoid para saída binária
        return 1.0 / (1.0 + np.exp(-o))

    def evaluate(self, individual):
        """
        Fitness = -Binary Cross-Entropy.
        Quanto menor o erro, maior o fitness.
        """
        preds = self.forward(individual, self.X)
        eps = 1e-8
        loss = -(self.y * np.log(preds + eps) +
                 (1 - self.y) * np.log(1 - preds + eps)).mean()
        return -loss

    # ---------- Mutação / Evolução ----------

    def mutate(self, individual):
        return [
            w + self.rng.normal(0, self.mutation_rate, size=w.shape)
            for w in individual
        ]

    def evolve_one_generation(self, elite_frac=0.2):
        # fitness na população atual
        fitness = np.array([self.evaluate(ind) for ind in self.population])

        n_elite = max(2, int(self.pop_size * elite_frac))
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [self.population[i] for i in elite_idx]

        # nova população: elites + mutantes
        new_pop = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = elites[self.rng.integers(0, len(elites))]
            new_pop.append(self.mutate(parent))

        self.population = new_pop

        # fitness na nova população
        new_fitness = np.array([self.evaluate(ind) for ind in self.population])
        best_idx = int(np.argmax(new_fitness))
        best = self.population[best_idx]
        best_fit = float(new_fitness[best_idx])

        return best, best_idx, self.population, best_fit
