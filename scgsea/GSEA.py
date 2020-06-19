import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union

import gseapy as gp
import numpy as np
import pandas as pd
import seaborn as sns

from gsea.exceptions import SinglePopulationError
from cellforest.CellORM import CellForest
from cellforest.Counts import Counts


class GSEA:
    sns.set(color_codes=True)
    # TODO: is this a mortal sin?
    logger = logging.getLogger("GSEA")
    GENE_SET_REF_DIR = Path('PATH_NOT_SET')
    GENE_SET_FILENAMES = {"<GENE_SET_NAME_HERE>": "<GENE_SET_FILENAME_HERE"}
    GSEA_OUTPUT_FILENAME = "gseapy.gsea.phenotype.report.csv"
    USE_CPUS = multiprocessing.cpu_count() - 2
    logger.info(f"using {USE_CPUS} cpus")
    DEFAULT_KWARGS = {
        "graph_num": 20,
        "max_size": 500,
        "method": "log2_ratio_of_classes",
        "min_size": 15,
        "no_plot": False,
        "permutation_type": "phenotype",
        "processes": USE_CPUS,
        # TODO: changed this and testing the impact
        "permutation_num": 10000,
        "seed": None,
        "verbose": True,
        "weighted_score_type": 1,
    }

    def __init__(self, orm: CellForest, process_name: str):
        self.orm = orm.at(process_name)
        self.gene_set = self.orm.spec[process_name]["gene_set"]
        self.gene_set_path = self.GENE_SET_REF_DIR / self.GENE_SET_FILENAMES[self.gene_set]
        self.process_name = process_name

    def run(self, **kwargs):
        pos_label = sorted(self.orm.meta["partition_code"].unique())[0]  # first alphabetically
        if "pos_label" in kwargs:
            pos_label = kwargs.pop("pos_label")
        self.run_static(self.orm, self.gene_set_path, self.process_name, pos_label, **kwargs)

    @property
    def results_df(self) -> pd.DataFrame:
        if not self.done:
            raise ValueError(
                f"No GSEA run exists in {self.output_path} for {self.gene_set}. Please use "
                "`GeneSetEnrichmentAnalyzer.run` to create one."
            )
        results_df = pd.read_csv(self.output_path, index_col="Term")
        return results_df

    @property
    def output_path(self) -> Path:
        """Full output path of `.csv` file from `results_df`"""
        return self.orm.paths[self.process_name] / self.GSEA_OUTPUT_FILENAME

    @property
    def done(self) -> bool:
        return self.output_path.is_file()

    @staticmethod
    def run_static(
        orm: CellForest,
        gene_set_path: str,
        process_name: str,
        pos_label: str,
        shuffling: Optional[str] = None,
        shuffling_keep_class_balance: Optional[bool] = True,
        cell_labels: Optional[list] = None,
        counts: Optional[Counts] = None,
        **kwargs,
    ):
        """

        Args:
            orm: spec must use "gsea" or "gsea_bulk" key to specify process
                params and partitions across which GSEA compares populations.
                Any subsets and filters must also be specified here.
                e.g.:
                    spec = {...
                              "gsea": {
                                "alpha": 0.05,
                                "disease_name": "disease_1",
                                "partition": {"disease_state", },
                            }
            gene_set_path: path to `.gmt` file from MSigDB
            process_name: compliant with `ProcessSchemaSC`, e.g. 'gsea_bulk'
            pos_label: label for which enrichment will result in positive NES
            shuffling: label shuffling methods to compare observed effects to
                results. Options:
                    "cell": randomly shuffle cell labels
                    "sample": shuffle labels across entire samples, keeping
                        labels consistent within a given sample
                    None: no shuffling - used to get original result
            shuffling_keep_class_balance: keep original class imbalance or rest
                to 50/50
            cell_labels: labels to override calculated ones
            counts: counts matrix to override counts from orm

        # TODO: add variable names to GSEA input file after encoding
        """
        GSEA.logger.info(f"Running GSEA on {orm.counts.shape}")
        if counts is not None:
            counts = counts.copy()
        else:
            counts = orm.counts.copy()
        if not cell_labels:
            counts, cell_labels = GSEA.get_cell_labels(
                orm.meta, counts, pos_label, shuffling, shuffling_keep_class_balance
            )
        print(cell_labels[:10])
        GSEA.gsea_wrapper(
            counts=counts,
            cell_labels=cell_labels,
            gene_set_gmt=str(gene_set_path),
            output_dir=orm.paths[process_name],
            **kwargs,
        )
        GSEA.logger.info(f"GSEA RESULTS SAVED TO {orm.paths[process_name]}")

    @staticmethod
    def gsea_wrapper(
        counts: Counts,
        cell_labels: Union[List[int], np.ndarray],
        gene_set_gmt: str,
        output_dir: Union[Path, str],
        **kwargs,
    ):
        """
        Uncoupled from `CellORM` infrastructure
        Args:
            counts: counts
            cell_labels:
            gene_set_gmt:
            output_dir:
            **kwargs:
        """
        gct_df = counts.to_df().T
        gct_df.insert(0, "Description", "None")
        gct_df.index = gct_df.index.rename("NAME")
        kwargs = {
            **GSEA.DEFAULT_KWARGS,
            "data": gct_df,
            "gene_sets": gene_set_gmt,
            "cls": cell_labels,
            "outdir": str(output_dir),
            **kwargs,
        }
        print(kwargs["permutation_num"])
        try:
            gp.gsea(**kwargs)
        except IndexError:
            raise SinglePopulationError(cell_labels[0])

    @staticmethod
    def get_cell_labels(
        meta, counts, pos_label, shuffling=None, shuffling_keep_class_balance=True, prob_pos_override=None
    ):
        cell_labels = meta["partition_code"].tolist()
        label_classes = set(cell_labels)
        if len(label_classes) == 1:
            raise ValueError(f"Dataset must contain two labels classes, but contains only one: {pos_label}")
        (neg_label,) = label_classes.difference({pos_label})
        GSEA.logger.info(f"{pos_label} vs {neg_label}")

        # First label must be positive label. Rearrange counts and labels accordingly
        i_pos = np.where(np.array(cell_labels) == pos_label)[0][0]
        temp = cell_labels[i_pos]
        cell_labels[i_pos] = cell_labels[0]
        cell_labels[0] = temp
        index = counts.index.copy()
        temp = index[0]
        index[0] = index[i_pos]
        index[i_pos] = temp
        counts = counts[index]
        return counts, cell_labels
