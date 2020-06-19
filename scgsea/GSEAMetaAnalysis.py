import logging
import re
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from gsea.GSEA import GSEA
from gsea.GSEAGroup import GSEAGroup
from cellforest.CellORM import CellForest


class GSEAMetaAnalysis:
    COLORS = (
        (0, 0.75, 0.75),
        (0.75, 0, 0.75),
        (0.75, 0.75, 0),
        (0, 0, 0.75),
        (0, 0.75, 0),
        (0.75, 0, 0),
        (0, 0.25, 0.25),
        (0.25, 0, 0.25),
        (0.25, 0.25, 0),
        (0, 0, 0.25),
        (0, 0.25, 0),
        (0.25, 0, 0),
    )
    HIERARCHICAL_VARS = [
        "experiment",
        "disease_name",
    ]

    @staticmethod
    def scatter_tornado(results: List[pd.DataFrame], max_opacity=0.5, title: Optional[str] = None, **kwargs):
        # TODO: refactor to separate from processing to prevent having to reprocess due to plotting error
        plt.figure(figsize=(16, 12))
        if title:
            plt.title(title)

        # TODO: this breaks single run without grouping - either unify DataFrame format between ProcessMethodsSC and BatchMethodsSC
        # if isinstance(results, pd.DataFrame):
        #     # results_df_iters = [ProcessMethodsSC.gsea_bulk(orm, **kwargs).results_df for _ in range(n_repeat)]
        #     df_fdr = pd.concat([df['fdr'] for df in results_df_iters], axis=1, sort=True).T.reset_index(drop=True)
        #     df_opacity = df_fdr.applymap(lambda x: 0.5 * (1 - x))
        #     df_nes = pd.concat([df['nes'] for df in results_df_iters], axis=1, sort=True).T.reset_index(drop=True)
        #     GSEAMetaAnalysis._scatter_tornado_single(df_nes, df_opacity)
        legend_elements = []
        cols = filter(lambda x: x.startswith("fdr"), results[0].columns)
        grp_labels = list(map(lambda x: x[4:], cols))
        for i, label in enumerate(grp_labels):
            fdr_col = "_".join(("fdr", label))
            nes_col = "_".join(("nes", label))
            if nes_col not in results[0].columns:
                logging.warning(f"{label} was not run due to an error and will not be plotted")
                continue
            df_nes = pd.concat([df[nes_col] for df in results], axis=1, sort=True).T.reset_index(drop=True)
            df_fdr = pd.concat([df[fdr_col] for df in results], axis=1, sort=True).T.reset_index(drop=True)
            df_opacity = df_fdr.applymap(lambda x: max_opacity * (1 - x))
            rgb = GSEAMetaAnalysis.rgb_cycler(i)
            GSEAMetaAnalysis._scatter_tornado_single(df_nes, df_opacity, rgb, **kwargs)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=(*rgb, max_opacity))
            )
        plt.legend(handles=legend_elements)

    @staticmethod
    def _scatter_tornado_single(df_nes, df_opacity, rgb=(0.5, 0.5, 0.5), **kwargs):
        for i, (name, y) in enumerate(df_nes.iteritems()):
            x = np.random.uniform(i - 0.15, i + 0.15, len(y))
            s = df_opacity[name]
            rgb_arr = np.tile(rgb, (len(df_nes), 1))
            rgba = np.concatenate([rgb_arr, np.expand_dims(s.values, axis=1)], axis=1)
            plt.scatter(x=x, y=y, c=rgba, **kwargs)
        plt.xticks(range(len(df_nes.columns)), df_nes.columns, rotation=90)
        plt.ylabel("Normalized Expression Score (NES)")

    @staticmethod
    def tornado(
        orm: CellForest,
        process_name: str,
        grouping_vars: Optional[Union[str, list, set, tuple]] = None,
        facet_vars: Optional[Union[str, list, set, tuple]] = None,
        labels: Union[list, np.ndarray] = None,
        n_hypotheses: int = 1,
        **kwargs,
    ) -> Union[plt.Axes, List[plt.Axes]]:
        """
        Creates a horizontal bar plot (tornado) of GSEA NES values categorized
        by gene set specified in `orm.spec` after GSEA has been run.

        Args:
            orm:
            process_name:
            grouping_vars: a separate colored set of bars is created for each
                set of values of these variables
            facet_vars: a separate plot is created for each set of values for
                these variables
            labels: labels for legend
            n_hypotheses: additional hypothesis correction *outside* of this
                plotting process (i.e., hypothesis correction for
                `grouping_vars` and `facet_vars` is already included, so this
                should generally only be adjusted when multiple calls are being
                made to this method
            **kwargs:

        Returns:
            ax or list of ax:
        """
        if facet_vars:
            for _, grp_orm in orm.groupby(facet_vars):
                if grouping_vars:
                    GSEAMetaAnalysis.tornado_multi(grp_orm, process_name, grouping_vars, labels, n_hypotheses)
                else:
                    GSEAMetaAnalysis.tornado_single(orm, process_name, **kwargs)
        elif grouping_vars:
            return GSEAMetaAnalysis.tornado_multi(orm, process_name, grouping_vars, labels, **kwargs)
        else:
            return GSEAMetaAnalysis.tornado_single(orm, process_name, **kwargs)

    @staticmethod
    def tornado_single(
        orm: CellForest, process_name: str, n_hypotheses: int = 1, rgb: tuple = (0, 0.75, 0.75), figsize: tuple = (15, 15)
    ) -> plt.Axes:
        gsea = GSEA(orm, process_name)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        x = range(len(gsea.results_df))
        rgba = np.zeros((len(gsea.results_df), 4))
        rgba[:, 0:3] = rgb
        corrected_fdr = (gsea.results_df["fdr"].values * n_hypotheses).clip(max=1)
        rgba[:, 3] = GSEAMetaAnalysis.pval_to_opacity(corrected_fdr)
        ax.barh(x, gsea.results_df["nes"], color=rgba)
        ax.set_yticks(x)
        ax.set_yticklabels(list(gsea.results_df.index))
        ax.set_xlabel("Normalized Expression Score")
        ax.invert_yaxis()
        return ax

    @staticmethod
    def tornado_multi(
        orm: CellForest,
        process_name: str,
        grouping_vars: Optional[Union[str, list, set, tuple]] = None,
        labels: Union[list, tuple] = None,
        n_hypotheses: int = 1,
    ) -> plt.Axes:
        """
        Like `plot_tornado_single`, except there with a colored set of bars for
        each value corresponding to `grouping_vars`.
        Args:
            orm:
            process_name:
            grouping_vars:
            labels:
            n_hypotheses:

        Returns:
            ax:
        """

        gsea_grp = GSEAGroup(orm, grouping_vars, process_name)
        results_df = gsea_grp.group_results_df
        group_names = [re.sub("fdr_*", "", col) for col in results_df.columns if re.compile("fdr_*").match(col)]
        if not gsea_grp.groups == tuple(group_names):
            raise ValueError()
        n_groups = len(gsea_grp.groups)
        figsize = (8, len(results_df) / 3 * (1 + 0.1 * n_groups))
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        x_max = n_groups * len(results_df)
        n_hypotheses *= n_groups
        for i, grp_vals in enumerate(gsea_grp.groups):
            if not isinstance(grp_vals, str):
                grp_vals = "_".join(grp_vals)
            fdr_col = f"fdr_{grp_vals}"
            nes_col = f"nes_{grp_vals}"
            x = range(i, x_max + i - n_groups + 1, n_groups)
            rgba = np.zeros((len(results_df), 4))
            rgba[:, 0:3] = GSEAMetaAnalysis.COLORS[i]
            corrected_fdr = (results_df[fdr_col].values * n_hypotheses).clip(max=1)
            rgba[:, 3] = GSEAMetaAnalysis.pval_to_opacity(corrected_fdr)
            ax.barh(x, results_df[nes_col], color=rgba)
            if round(n_groups) == i:
                ax.set_yticks(x)
            if i == 0:
                x_markers = x
        if labels is not None:
            color = GSEAMetaAnalysis.COLORS[i]
            legend = [(Line2D([0], [0], color=color, lw=4), label) for i, label in enumerate(labels)]
            ax.legend(*list(zip(*legend)), edgecolor="inherit", facecolor="white")
        (ax_min, ax_max, _, _) = ax.axis()

        ax.hlines(x_markers, ax_min, ax_max, lw=0.1, alpha=0.8)
        ax.hlines(x_markers, ax_min, ax_max, lw=0.1, alpha=0.8)
        ax.set_yticklabels(list(results_df.index))
        ax.set_yticks(np.arange(len(results_df)) * n_groups + n_groups / 2 - 1)
        ax.set_xlabel("Normalized Expression Score")
        ax.invert_yaxis()
        return ax

    @staticmethod
    def tornado_faceted(
        orm: CellForest,
        process_name: str,
        grouping_vars: Optional[Union[str, list, set, tuple]],
        facet_vars: Union[str, list, set, tuple],
        n_hypotheses: int = 1,
    ) -> List[plt.Axes]:
        """
        Like `plot_multi_tornado` except a new plot is created for each set
        of values corresponding to `facet_vars`"""
        # TODO: orm groupby for facet_vars then call plot_multi_tornado?
        axes = []
        gsea_facet_grps = GSEAGroup(orm, facet_vars, process_name)
        n_hypotheses *= len(gsea_facet_grps.orm_lookup)
        for grp_vals, orm in gsea_facet_grps.orm_lookup.items():
            ax = GSEAMetaAnalysis.tornado_multi(orm, process_name, grouping_vars)
            ax.set_title(str(grp_vals))
            axes.append(ax)
        return axes

    @staticmethod
    def heatmap(
        orm: CellForest, gene_set_name: str, min_cells: int, gsea: Optional[GSEA] = None, **kwargs
    ) -> sns.matrix.ClusterGrid:
        """
        Heatmap of `GSEAMetaAnalysis.samples_by_gene_set_expr` with `sample_id`
        and other metadata in plot labels.
        """
        sample_expr = GSEAMetaAnalysis.samples_by_gene_set_expr(orm, gene_set_name, min_cells, gsea)
        return GSEAMetaAnalysis._plot_heatmap(sample_expr, **kwargs)

    @staticmethod
    def samples_by_gene_set_expr(
        orm: CellForest, gene_set: Union[str, List[str]], min_cells: int, gsea: Optional[GSEA] = None
    ) -> pd.DataFrame:
        """
        Matrix of average expression over a sample for a given gene set. If
        cell level `subset`ing is desired (e.g., by `cell_id`), batch this
        after running an `orm.groupby` to generate a separate plot for each
        cluster.
        Args:
            orm: root level configs will be used to get cells
            gsea: used only to get gene names from gene set which are present
                in data
            gene_set: either string name of key for .gmt in
                `GSEA.GENE_SET_FILENAMES`, or list of strings which are gene
                names that match gene names from `orm.counts.genes`
            min_cells: samples with fewer cells will be excluded from output

        Returns:
            sample_expr_gene_set_sorted: [samples x genes], ordered by
                `get_sample_hier_counts`
        """
        if isinstance(gene_set, str):
            genes = gsea.results_df["genes"].loc[gene_set].split(";")
        else:
            genes = gene_set
        counts = orm.counts[:, genes].to_df()
        counts.insert(1, "sample_id", orm.meta["sample_id"].astype("int64").tolist())
        good_samples = GSEAMetaAnalysis._get_good_samples(counts["sample_id"], min_cells)
        hier_counts = GSEAMetaAnalysis.get_sample_hier_counts(orm, good_samples)
        # TODO: avoid hardcoding `5`?
        sample_index = hier_counts.index.get_level_values(5)
        good_samples_sorted = sample_index[sample_index.isin(good_samples)].tolist()
        sample_expr_gene_set = counts.groupby("sample_id").agg(np.mean)
        sample_expr_gene_set = sample_expr_gene_set.loc[good_samples]
        sample_expr_gene_set_sorted = sample_expr_gene_set.loc[good_samples_sorted]
        sample_meta = orm.meta.groupby("sample_id").agg(min).loc[sample_expr_gene_set_sorted.index].reset_index()
        sample_meta = sample_meta[GSEAMetaAnalysis.HIERARCHICAL_VARS[::-1]]
        sample_strs = sample_meta.apply(lambda x: "-".join(map(str, x.tolist())), axis=1)
        sample_expr_gene_set_sorted["sample_str"] = sample_strs.values
        sample_expr_gene_set_sorted = sample_expr_gene_set_sorted.set_index("sample_str")
        return sample_expr_gene_set_sorted

    @staticmethod
    def get_sample_hier_counts(orm: CellForest, good_samples: Union[List[int], np.ndarray, pd.Series] = None) -> pd.Series:
        """
        Groupby and aggregation on variables in order of
        `GSEAMetaAnalysis.HIERARCHICAL_VARS`. This orders samples hierarchically
        and the aggregation gets the cell count per sample.
        Args:
            orm: used for `orm.meta` to do groupby
            good_samples: if specified, samples not included will be filtered out

        Returns:
            hier_counts:
        """
        # TODO: get string out of here for labels on heatmap
        if good_samples is not None:
            meta_good = orm.meta[orm.meta.sample_id.isin(good_samples)]
        else:
            meta_good = orm.meta
        meta_good = meta_good[GSEAMetaAnalysis.HIERARCHICAL_VARS]
        hier_grp = meta_good.groupby(GSEAMetaAnalysis.HIERARCHICAL_VARS)
        hier_counts = hier_grp.agg(len)
        return hier_counts

    @staticmethod
    def _plot_heatmap(sample_expr: pd.DataFrame, size: Union[int, float] = 1, **kwargs) -> sns.matrix.ClusterGrid:
        figsize = (np.array(sample_expr.values.T.shape) * size / 3).tolist()
        default_kwargs = {"xticklabels": True, "yticklabels": True, "figsize": figsize}
        default_kwargs.update(kwargs)
        return sns.clustermap(sample_expr, **default_kwargs)

    @staticmethod
    def _get_good_samples(cell_samples: pd.DataFrame, min_cells: int) -> List[int]:
        # TODO incorporate this and add as param to infra
        sample_grp = cell_samples.groupby(cell_samples)
        sample_counts = sample_grp.agg(len)
        good_samples = sample_counts[sample_counts > min_cells].index.tolist()
        return good_samples

    @staticmethod
    def _multiple_hypothesis_correction(fdr):
        raise NotImplementedError()

    @staticmethod
    def rgb_cycler(x):
        r = 0.5 + 0.5 * np.sin(x)
        g = 0.5 + 0.5 * np.sin(x + 2 * np.pi / 3)
        b = 0.5 + 0.5 * np.sin(x + 4 * np.pi / 3)
        return r, g, b


def _pval_to_opacity_worker(pval: float, alpha: float = 0.05, nonsig_max: float = 0.25) -> float:
    """
    Converts p-values or fdr to opacity, with opacity cliff at `alpha` where
    opacity is 1 below `alpha`, and linearly interpolated between `nonsig_max`
    and 0 from `alpha` to 1.
    Args:
        pval: p-value or fdr to convert to opacity
        alpha: significance threshold for opacity cliff
        nonsig_max: maximum opacity for non-significant results

    Returns:

    """
    if pval < alpha:
        return 1
    else:
        return nonsig_max * (1 - pval)


GSEAMetaAnalysis.pval_to_opacity = np.vectorize(_pval_to_opacity_worker, otypes=[np.float])
