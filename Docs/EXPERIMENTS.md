# Investigação estendida (prévia da dissertação)

Guia rápido dos experimentos **exp5–30**, que estendem o pacote do paper (exp1–4, ver
[`README.md`](../README.md)). A narrativa completa, com figuras e explicações, está em
**[`relatorios/experiment_report.html`](relatorios/experiment_report.html)** — abra no
navegador; o botão **⬇ Baixar PDF** imprime para PDF.

> **Pergunta → mistério → causa → recompensa.** Qual métrica usar? O espaço de pesos
> organiza a fitness? Por que não? (simetrias) Como a geometria revela o algoritmo de busca?

---

## Dois ambientes (um único projeto pixi, da raiz do repo)

| ambiente | flag | para quê | numpy |
|---|---|---|---|
| **viz** (default) | `-e viz` | UMAP, análises, figuras, app | 1.26 |
| **evo** | `-e evo` | gerar runs de ES (jax + brax + evosax) → `Experiments/runs/` | 2.x |

`pixi install` resolve os dois. A lib `Experiments/src` é pacote editável nos dois envs.

## Rodar análises (ambiente viz)

```bash
# fundação
pixi run -e viz python Experiments/scripts/05_umap_param_search.py
pixi run -e viz python Experiments/scripts/11_evolution_summary.py
# métricas
pixi run -e viz python Experiments/scripts/08_metric_comparison.py --force-rerun
pixi run -e viz python Experiments/scripts/09_metric_validation.py
pixi run -e viz python Experiments/scripts/12_metric_comparison_working.py
pixi run -e viz python Experiments/scripts/13_metric_comparison_fitness.py
# espaço de pesos vs comportamental
pixi run -e viz python Experiments/scripts/14_behavior_vs_weight_space.py
pixi run -e viz python Experiments/scripts/15_fitness_aware_descriptor.py
pixi run -e viz python Experiments/scripts/16_3d_embedding.py
# simetrias (clímax)
pixi run -e viz python Experiments/scripts/21_functional_equivalence.py
pixi run -e viz python Experiments/scripts/23_graph_embedding_demo.py
pixi run -e viz python Experiments/scripts/29_signed_graph_embedding.py   # corrige a cegueira ao sinal do exp23
# geometria da busca
pixi run -e viz python Experiments/scripts/17_joint_es_comparison.py
pixi run -e viz python Experiments/scripts/18_joint_es_extended.py
pixi run -e viz python Experiments/scripts/19_joint_es_moons.py
pixi run -e viz python Experiments/scripts/20_temporal_evolution_gif.py
pixi run -e viz python Experiments/scripts/22_search_dynamics.py
pixi run -e viz python Experiments/scripts/24_cross_task_signatures.py
pixi run -e viz python Experiments/scripts/30_control_task_signatures.py  # estende exp24 p/ CartPole + HalfCheetah
# aplicações propostas + ablação de confound
pixi run -e viz python Experiments/scripts/26_premature_convergence.py
pixi run -e viz python Experiments/scripts/27_offspring_allocation.py
pixi run -e viz python Experiments/scripts/28_controlled_confound.py
```

## Gerar os runs de neuroevolução (ambiente evo)

```bash
pixi run -e evo es-sanity                                                            # valida os ES (ótimo = 0)
pixi run -e evo python Experiments/src/neuroevolve_brax.py --seed 42                 # HalfCheetah (GA)
pixi run -e evo python Experiments/scripts/neuroevolve_evosax.py --algo cma_es --seed 42
pixi run -e evo python Experiments/scripts/moons_evosax.py --all_algos --seed 42 --task moons
pixi run -e evo python Experiments/scripts/extract_behavior.py --seed 42
```

Atalhos no env evo: `es-sanity`, `evolve-halfcheetah`, `evolve`, `moons`, `cartpole`,
`extract-behavior`. Tarefas de classificação em `--task`: `moons`, `circles`, `blobs`, `xor`
(datasets pré-gerados em `Experiments/runs/{task}_data_seed*.npz`).

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
| 3. Causa (simetrias) | 21, 23, 29 |
| 4. Geometria da busca | 17, 18, 19, 24, 20, 22, 30 |
| 5. Aplicações + confound | 26, 27, 28 |
| Apêndice | 2, 4, 6, 7 |

## Tempo aproximado

| etapa | tempo |
|---|---|
| análises pixi (exp8–24, com cache) | minutos |
| runs de classificação (4 tasks × 3 seeds, evosax) | ~1 min |
| runs HalfCheetah evosax (por algo/seed, ep=300) | alguns min |
| sanity-check dos ES | ~30 s |
