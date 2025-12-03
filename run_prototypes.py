import numpy as np
from ne_moons import NeuroEvoMoons
from project_weights import project_all_generations
from visualizations import (
    plot_population_generations_global,
    plot_best_trajectory_global,
    plot_centroid_flow,
)

ne = NeuroEvoMoons()

num_generations = 30

populations_per_gen = []
best_global_indices = []  # índices globais no embedding

all_individuals_flat = []  # só pra construir índice global manualmente
all_fitness = []
gen_labels_tmp = []

for gen in range(num_generations):
    print(f"Geração {gen}...")

    best, best_idx, population, best_fit = ne.evolve_one_generation()

    # guarda população da geração g
    populations_per_gen.append(population)

    # registra índices globais para a linhagem do melhor
    # (posição do indivíduo best_idx dentro da "lista global" concatenada)
    offset = len(all_individuals_flat)
    # registra fitness da população antes de iterar para manter ordem consistente
    fitness_pop = np.array([ne.evaluate(ind) for ind in population])

    for i, ind in enumerate(population):
        all_individuals_flat.append(ne.flatten(ind))
        all_fitness.append(float(fitness_pop[i]))
        gen_labels_tmp.append(gen)
        if i == best_idx:
            best_global_indices.append(offset + i)

# agora projetamos TUDO junto com UMAP global
all_vectors = np.array(all_individuals_flat)
gen_labels_tmp = np.array(gen_labels_tmp)

from umap import UMAP
reducer = UMAP(
    n_neighbors=30,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    random_state=42,
)
embedding = reducer.fit_transform(all_vectors)

# plota população por geração, trajetória do melhor e fluxo de centróides
plot_population_generations_global(embedding, gen_labels_tmp)
plot_best_trajectory_global(embedding, best_global_indices)
plot_centroid_flow(embedding, gen_labels_tmp)

from visualizations import plot_population_colored_by_fitness, plot_fitness_over_generations

plot_population_colored_by_fitness(embedding, np.array(all_fitness), gen_labels_tmp)
plot_fitness_over_generations(np.array(all_fitness), gen_labels_tmp)

print("Figuras salvas em ./figures")
