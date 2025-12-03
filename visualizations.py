import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figures", exist_ok=True)


def plot_population_generations_global(embedding, gen_labels):
    plt.figure(figsize=(8, 6))
    gens = np.unique(gen_labels)
    for g in gens:
        pts = embedding[gen_labels == g]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"G{g}", alpha=0.6, s=25)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("UMAP – População por Geração (embedding global)")
    plt.tight_layout()
    plt.savefig("figures/population_generations_global.png", dpi=300)
    plt.close()


def plot_best_trajectory_global(embedding, best_global_indices):
    traj = embedding[best_global_indices]

    plt.figure(figsize=(7, 5))
    plt.plot(traj[:, 0], traj[:, 1], "-o", linewidth=2)
    for i, (x, y) in enumerate(traj):
        plt.text(x, y, f"G{i}", fontsize=9)
    plt.title("Trajetória do Melhor Indivíduo (embedding global)")
    plt.tight_layout()
    plt.savefig("figures/best_trajectory_global.png", dpi=300)
    plt.close()


def plot_centroid_flow(embedding, gen_labels):
    gens = np.unique(gen_labels)
    centroids = []
    for g in gens:
        pts = embedding[gen_labels == g]
        centroids.append(pts.mean(axis=0))
    centroids = np.array(centroids)

    plt.figure(figsize=(7, 5))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=60)
    for i, (x, y) in enumerate(centroids):
        plt.text(x, y, f"G{i}", fontsize=9)

    # desenha setas entre centróides consecutivos
    for i in range(len(centroids) - 1):
        x0, y0 = centroids[i]
        x1, y1 = centroids[i + 1]
        plt.arrow(
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            length_includes_head=True,
            head_width=0.1,
            alpha=0.8,
        )

    plt.title("Fluxo da População no Espaço de Pesos (centróides por geração)")
    plt.tight_layout()
    plt.savefig("figures/centroid_flow.png", dpi=300)
    plt.close()


def plot_population_colored_by_fitness(embedding, fitness, gen_labels):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=fitness,
        cmap="viridis",
        s=28,
        alpha=0.85,
    )
    plt.colorbar(sc, label="Fitness")
    plt.title("UMAP – Espaço de Pesos Colorido por Fitness")
    plt.tight_layout()
    plt.savefig("figures/population_by_fitness.png", dpi=300)
    plt.close()


def plot_fitness_over_generations(fitness, gen_labels):
    plt.figure(figsize=(8, 6))
    gens = np.unique(gen_labels)
    avg = [np.mean([fitness[i] for i in range(len(fitness)) if gen_labels[i] == g])
           for g in gens]
    std = [np.std([fitness[i] for i in range(len(fitness)) if gen_labels[i] == g])
           for g in gens]

    plt.errorbar(gens, avg, yerr=std, fmt="-o", capsize=4)
    plt.xlabel("Geração")
    plt.ylabel("Fitness médio ± desvio")
    plt.title("Evolução do Fitness por Geração")
    plt.tight_layout()
    plt.savefig("figures/fitness_by_generation.png", dpi=300)
    plt.close()
