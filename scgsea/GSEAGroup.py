import logging
from typing import Union

import pandas as pd

from cellforest.CellORM import CellForest
from cellforest.ProcessMethodsSC import ProcessMethodsSC


class GSEAGroup:
    """

    Args:
        orm: the base orm which is not subset for each constituent group. Includes `partition`
        grouping_vars: the vars over which the constituent groups vary (e.g. {'cluster_id', 'disease_state'})
            ??? is this not a redundant version of partitions? Maybe not redundant because `partition` could be for any process?
    """

    def __init__(self, orm: CellForest, grouping_vars: Union[str, list, set, tuple], process_name: str):
        self._orm_lookup = {grp_vals: grp_orm for (grp_vals, grp_orm) in orm.groupby(grouping_vars)}
        self.groups = list(self._orm_lookup.keys())
        self.results_lookup = dict()
        self._group_results_df = None

    def run_grp(self, name, **kwargs):
        try:
            self.results_lookup[name] = ProcessMethodsSC.gsea_bulk(self._orm_lookup[name], **kwargs).results_df
        except Exception as e:
            logging.warning(f"Group {name} not run due to error:")
            logging.error(e)

    @property
    def group_results_df(self) -> pd.DataFrame:
        """See `_get_group_results_df`"""
        if self._group_results_df is None:
            self._group_results_df = self._get_group_results_df()
        return self._group_results_df

    def _get_group_results_df(self) -> pd.DataFrame:
        """
        Combine 'nes' and 'fdr' values from individual `GSEA` runs. The columns
        are named via the convention: {fdr, nes}_var1_var2..., where var1 and
        var2 are the values of the values used for grouping
        Returns:
            group_df:
        """
        group_df = None
        genes_df = None
        gene_colnames = ["genes", "ledge_genes"]
        for grp_vals, full_results_df in self.results_lookup.items():
            results_df = full_results_df[["fdr", "nes"]]
            if isinstance(grp_vals, (list, tuple)):
                grp_vals = "_".join(map(str, grp_vals))
            results_df = results_df.add_suffix(f"_{grp_vals}")
            if group_df is None:
                group_df = results_df
                genes_df = full_results_df[gene_colnames]
            else:
                group_df = pd.concat([group_df, results_df], axis=1, sort=True)
        group_df = pd.concat([group_df, genes_df], axis=1, sort=True)
        ordered_cols = sorted(list(set(group_df.columns).difference({"genes", "ledge_genes"}))) + gene_colnames
        return group_df[ordered_cols]
