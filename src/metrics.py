import jax
import jax.numpy as jnp
import qdax.core.containers.mapelites_repertoire
from qdax.types import Metrics

import csv
import os
from collections.abc import Iterable
from typing import Any

from .utils import transpose_dict_of_list


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: list[str]) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        if not os.path.exists(self._filename):
            with open(self._filename, "w") as file:
                writer = csv.DictWriter(file, fieldnames=self._header)
                # write the header
                writer.writeheader()

    def log(self, metrics: Iterable[dict[str, Any]]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in raws
            writer.writerows(metrics)


def qd_metrics(
    repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
    qd_offset: float,
) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)
    min_fitness = jnp.min(repertoire.fitnesses, initial=max_fitness, where=~repertoire_empty)
    mean_fitness = jnp.mean(repertoire.fitnesses, where=~repertoire_empty)

    return {
        'qd_score': qd_score,
        'max_fitness': max_fitness,
        'coverage': coverage,
        'min_fitness': min_fitness,
        'mean_fitness': mean_fitness,
    }


def get_logged_metrics(
    main_metrics: dict[str, Iterable[int | float | jax.Array]],
    other_metrics: dict[str, Iterable[int | float | jax.Array]],
    header: Iterable[str],
) -> tuple[str, list[dict[str, int | float | jax.Array]], dict[str, Any]]:
    main_metric_keys = tuple(main_metrics.keys())
    other_metric_keys = tuple(other_metrics.keys())

    metrics = main_metrics | other_metrics

    metrics_list = transpose_dict_of_list(metrics)
    latest_metrics = metrics_list[-1]

    wandb_metrics: dict[str, int | float | jax.Array] = {}
    for key in main_metric_keys:
        wandb_metrics[f'main/{key}'] = latest_metrics[key]
    for key in other_metric_keys:
        wandb_metrics[f'others/{key}'] = latest_metrics[key]

    log_metrics: list[str] = []
    for key in header:
        metric = latest_metrics[key]
        if (
            (isinstance(metric, jax.Array) and isinstance(metric.item(), float))
            or isinstance(metric, float)
        ) and metric < 10000.0:
            log_metrics.append(f'{metric:.5}')
        else:
            log_metrics.append(str(metric))

    return ' '.join(log_metrics), metrics_list, wandb_metrics
