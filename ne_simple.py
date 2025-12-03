import numpy as np

class SimpleNeuroevolution:
    def __init__(self, pop_size=40, input_dim=20, hidden_dim=32, output_dim=10, mutation_rate=0.05):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.shapes = [
            (input_dim, hidden_dim),
            (hidden_dim, output_dim)
        ]

        self.population = [self.random_individual() for _ in range(pop_size)]

        # XNOR dataset (expandido para 20 dims)
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.X = np.repeat(self.X, 10, axis=1)  # repeat to reach 20 dims
        self.y = np.array([1,0,0,1])

    def random_individual(self):
        return [np.random.randn(*shape) * 0.5 for shape in self.shapes]

    def flatten(self, individual):
        return np.concatenate([w.flatten() for w in individual])

    def forward(self, individual, x):
        w1, w2 = individual
        h = np.tanh(x @ w1)
        o = np.tanh(h @ w2)
        return o.mean()

    def evaluate(self, individual):
        preds = np.array([self.forward(individual, x.reshape(1, -1)) for x in self.X])
        loss = np.mean((preds - self.y)**2)
        return -loss   # maximize fitness

    def mutate(self, individual):
        new = []
        for w in individual:
            noise = np.random.randn(*w.shape) * self.mutation_rate
            new.append(w + noise)
        return new

    def evolve_one_generation(self):
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        elite_idx = np.argsort(fitness)[-5:]
        elites = [self.population[i] for i in elite_idx]

        new_pop = []
        for _ in range(self.pop_size):
            p = elites[np.random.randint(0, len(elites))]
            new_pop.append(self.mutate(p))

        self.population = new_pop
        return elites[0], self.population
