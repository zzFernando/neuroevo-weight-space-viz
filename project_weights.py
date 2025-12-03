import numpy as np
import umap


def project_all_generations(populations_per_gen, flatten_fn, n_neighbors=30, min_dist=0.05):
    """
    populations_per_gen: lista de listas de indivíduos.
        Ex: [ [ind_0_G0, ind_1_G0, ...],
              [ind_0_G1, ind_1_G1, ...],
              ... ]
    flatten_fn: função que converte indivíduo em vetor 1D.
    Retorna:
        embedding: array (N_total, 2)
        gen_labels: array (N_total,) com o número da geração de cada ponto.
        index_slices: lista com fatias para cada geração
                      (útil se tu quiser separar depois).
    """
    all_vecs = []
    gen_labels = []
    index_slices = []

    start = 0
    for gen, pop in enumerate(populations_per_gen):
        for ind in pop:
            all_vecs.append(flatten_fn(ind))
            gen_labels.append(gen)
        end = start + len(pop)
        index_slices.append(slice(start, end))
        start = end

    X = np.array(all_vecs)
    gen_labels = np.array(gen_labels)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(X)
    return embedding, gen_labels, index_slices
