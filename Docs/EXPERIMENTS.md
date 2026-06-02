# Investigação estendida (prévia da dissertação)

Guia rápido dos experimentos **exp5–25**, que estendem o pacote do paper (exp1–4, ver
[`README.md`](README.md)). A narrativa completa, com figuras e explicações, está em
**[`experiment_report.html`](experiment_report.html)** — abra no navegador; o botão
**⬇ Baixar PDF** imprime para PDF.

> **Pergunta → mistério → causa → recompensa.** Qual métrica usar? O espaço de pesos
> organiza a fitness? Por que não? (simetrias) Como a geometria revela o algoritmo de busca?

---

## Dois ambientes

| ambiente | onde | para quê |
|---|---|---|
| **pixi** | `supplementary/` | UMAP, análises, figuras (`pixi install`) |
| **JAX/Brax venv** | `brax/.venv` | gerar runs de ES (evosax + brax) → `runs/` |

## Rodar (ambiente pixi)

```bash
cd supplementary
# fundação
pixi run python experiments/05_umap_param_search.py
pixi run python experiments/11_evolution_summary.py
# métricas
pixi run python experiments/08_metric_comparison.py --force-rerun
pixi run python experiments/09_metric_validation.py
pixi run python experiments/12_metric_comparison_working.py
pixi run python experiments/13_metric_comparison_fitness.py
# espaço de pesos vs comportamental
pixi run python experiments/14_behavior_vs_weight_space.py
pixi run python experiments/15_fitness_aware_descriptor.py
pixi run python experiments/16_3d_embedding.py
# simetrias (clímax)
pixi run python experiments/21_functional_equivalence.py
pixi run python experiments/23_graph_embedding_demo.py
# geometria da busca
pixi run python experiments/17_joint_es_comparison.py
pixi run python experiments/18_joint_es_extended.py
pixi run python experiments/19_joint_es_moons.py
pixi run python experiments/20_temporal_evolution_gif.py
pixi run python experiments/22_search_dynamics.py
pixi run python experiments/24_cross_task_signatures.py
```

## Gerar os runs de neuroevolução (ambiente brax)

```bash
cd brax
.venv/bin/python es_sanity.py                                  # valida os ES (ótimo conhecido = 0)
.venv/bin/python neuroevolve_evosax.py --algo cma_es --seed 42 # HalfCheetah, 1 ES
.venv/bin/python moons_evosax.py --all_algos --seed 42 --task moons   # classificação 2D, 4 ES
.venv/bin/python extract_behavior.py --seed 42                 # descritores comportamentais
```

Tarefas de classificação disponíveis em `--task`: `moons`, `circles`, `blobs`, `xor`
(datasets pré-gerados com sklearn em `runs/{task}_data_seed*.npz`).

> ⚠️ **evosax minimiza** a fitness. Para maximizar uma recompensa `F`, passe `tell(-F)`.
> O `es_sanity.py` foi o que revelou um erro de sinal que invertia a otimização — rode-o
> sempre que mexer no harness de geração.

## Saída esperada (exemplo — Exp 24)

```
Fragmentation in UMAP (clusters) per task:
  Blobs/Moons/Circles/XOR:  Simple GA = 118–130   |  ES de distribuição = 3–20
→ figures/exp24_cross_task_umaps.png, exp24_cross_task_heatmap.png
```

## Mapa experimento → seção do relatório

| seção | experimentos |
|---|---|
| 0. Setup | 1, 3, 5, 25 (sanity), 11 |
| 1. Métrica | 8, 9, 10, 12, 13 |
| 2. Mistério (fitness) | 14, 15, 16 |
| 3. Causa (simetrias) | 21, 23 |
| 4. Geometria da busca | 17, 18, 19, 24, 20, 22 |
| Apêndice | 2, 4, 6, 7 |

## Tempo aproximado

| etapa | tempo |
|---|---|
| análises pixi (exp8–24, com cache) | minutos |
| runs de classificação (4 tasks × 3 seeds, evosax) | ~1 min |
| runs HalfCheetah evosax (por algo/seed, ep=300) | alguns min |
| sanity-check dos ES | ~30 s |
