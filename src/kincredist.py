from math import exp
import os
import pickle
from itertools import groupby
from typing import Optional

import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from numpy.lib.shape_base import row_stack
from xi_covutils.distances import calculate_distances

from crecredist import load_aln, load_domains, load_pdb_mappings, load_up_maps


def map_ungapped_to_gapped(
    gapped_seq:str
  ) -> dict[int, int]:
  ungapped = [i+1 for i, c  in enumerate(gapped_seq) if c!="-"]
  ungapped = {i+1:j for i, j in enumerate(ungapped)}
  return ungapped

# def collect_common_cre_residues(
#     aln,
#     domains,
#     up_chain_map,
#   ):


@click.command()
@click.option("--up", help="uniprot accession or reference MSA")
@click.option("--folder", help="data folder")
def calc_kin_cre_distances(up:str, folder:str):
  aln = load_aln(folder)
  up_chain_map = load_up_maps(folder)
  domains = load_domains(folder)
  up_ref = up
  rows = []
  map_to_msa = {}
  map_from_msa = {}
  for up in aln:
    if up not in up_chain_map:
      continue
    map_to_msa[up] = map_ungapped_to_gapped(aln[up])
    map_from_msa[up] = {v:k for k,v in map_to_msa[up].items()}
    for pdb_id, chain in up_chain_map[up]:
      rows.append(
        [
          up_ref,
          up,
          pdb_id,
          chain,
          domains[up]["cre"][0],
          domains[up]["cre"][1],
          domains[up]["kinase"][0],
          domains[up]["kinase"][1],
          map_to_msa[up][domains[up]["cre"][0]],
          map_to_msa[up][domains[up]["cre"][1]],
          map_to_msa[up][domains[up]["kinase"][0]],
          map_to_msa[up][domains[up]["kinase"][1]],
        ]
      )
  if not rows:
    print("There are no Sequences in with PDBs in this MSA.")
    exit(1)
  df = pd.DataFrame(
    rows,
    columns=[
      "up_ref",
      "up",
      "pdb",
      "chain",
      "cre_start",
      "cre_end",
      "kinase_start",
      "kinase_end",
      "cre_start_msa",
      "cre_end_msa",
      "kinase_start_msa",
      "kinase_end_msa",
    ]
  )
  common_cre_residues = (
    df["cre_start_msa"].max(),
    df["cre_end_msa"].min(),
  )
  common_kin_residues = (
    df["kinase_start_msa"].max(),
    df["kinase_end_msa"].min()
  )
  pdb_mappings = load_pdb_mappings(up_chain_map, folder)
  df = (
    df
      .assign(
        pdb_residues_cre =
          lambda x:
            x.apply(
              map_pdb_residues,
              pdb_mappings = pdb_mappings,
              region = common_cre_residues,
              from_msa_map = map_from_msa,
              axis =1
            )
      )
      .assign(
        all_none_cre =
          lambda x:
            x.pdb_residues_cre
              .apply(
                lambda y: all(v is None for v in y)
              )
      )
      .query("not all_none_cre")
      .drop("all_none_cre", axis=1)
      .assign(
        pdb_residues_kin =
          lambda x:
            x.apply(
              map_pdb_residues,
              pdb_mappings = pdb_mappings,
              region = common_kin_residues,
              from_msa_map = map_from_msa,
              axis =1
            )
      )
      .assign(
        all_none_kin =
          lambda x:
            x.pdb_residues_kin
              .apply(
                lambda y: all(v is None for v in y)
              )
      )
      .query("not all_none_kin")
      .drop("all_none_kin", axis=1)
      .assign(
        common_kin_res =
          lambda x:
            select_non_none_columns(x.pdb_residues_kin.to_list())
      )
      .assign(
        common_cre_res =
          lambda x:
            select_non_none_columns(x.pdb_residues_cre.to_list())
      )
  )
  distances = []
  for _, row in df.iterrows():
    cre_column = (
      "common_cre_res" if row["common_cre_res"] else "pdb_residues_cre"
    )
    kin_column = (
      "common_kin_res" if row["common_kin_res"] else "pdb_residues_kin"
    )
    atom_selector = ca_between_regions_selector(
      set(row[cre_column]),
      set(row[kin_column]),
      row["chain"]
    )
    pdb_file = os.path.join(
      folder,
      "input",
      "pdb",
      f"{row['pdb']}.pdb"
    )
    struct = PDBParser().get_structure(row["pdb"], pdb_file)
    models = struct.get_models()
    for mi, model in enumerate(models):
      dist_file = os.path.join(
        folder,
        "cache",
        f"{row['up_ref']}_{row['up']}_{row['pdb']}_{row['chain']}_{mi}_distances.pickle"
      )
      os.makedirs(
        os.path.dirname(dist_file),
        exist_ok=True
      )
      if os.path.exists(dist_file):
        with open(dist_file, "rb") as f_in:
          c_dist = pickle.load(f_in)
      else:
        c_dist = calculate_distances(model, atom_selector=atom_selector)
        with open(dist_file, "wb") as f_out:
          pickle.dump(c_dist, f_out)
      dist_df = (
        pd.DataFrame(
          c_dist,
          columns = ["chain1", "pos1", "chain2", "pos2", "distance"]
        )
        .drop(["chain1", "chain2"], axis=1)
      )
      distances.append(
        (
          row['up_ref'],
          row['up'],
          row['pdb'],
          row['chain'],
          mi,
          dist_df['pos1'],
          dist_df['pos2'],
          dist_df['distance']
        )
      )
  dist_df = (
    pd.DataFrame(
      distances,
      columns = [
        "up_ref",
        "up",
        "pdb",
        "chain",
        "model",
        "pos1",
        "pos2",
        "distance"
      ]
    )
    .explode(
      ["pos1", "pos2", "distance"]
    )
  )
  n_groups = len(dist_df[["up", "pdb", "model"]].drop_duplicates())
  min_width_per_group = 0.4
  min_total_width = 5
  width = min_total_width + n_groups * min_width_per_group + (1-exp(-n_groups))
  axes = (
    dist_df
      .drop(["pos1", "pos2"], axis=1)
      .assign(
        ID = lambda x:
          x[["up", "pdb", "model"]].agg(
            lambda a: "|".join([str(b) for b in a]),
            axis = 1
          )
      )
      .boxplot(
        column="distance",
        by = ["ID"],
        rot = 70,
        figsize = (width, 7)
      )
  )
  axes.set_title(f"Distances for Uniprot Reference MSA: {up_ref}")
  axes.set_ylabel("CA-CA residue distances (Å)")
  if df.loc[:, "common_cre_res"].iloc[0]:
    xlabel = "Using only alignable CRE residues"
  else:
    xlabel = "Not alignable CRE residues. Using all CRE residues."
  axes.set_xlabel(f"Uniprot | PDB id | Model #\n{xlabel}")
  plt.suptitle("")
  plt.tight_layout()
  out_distances_file = os.path.join(
    folder,
    "output",
    up_ref,
    f"cre_kinase_distances_{up_ref}.png"
  )
  plt.savefig(out_distances_file)


def ca_between_regions_selector(
    reg1: set[int],
    reg2: set[int],
    chain: str
  ):
  def selector(atom1: Atom, atom2: Atom) -> bool:
    if not atom1.id == "CA" or not atom2.id == "CA":
      return False
    res1 = atom1.parent
    res2 = atom2.parent
    if not isinstance(res1, Residue) or not isinstance(res2, Residue):
      return False
    chain1 = res1.parent
    chain2 = res2.parent
    if not isinstance(chain1, Chain) or not isinstance(chain2, Chain):
      return False
    if not chain1.id == chain or not chain2.id == chain:
      return False
    _, r1, _ = res1.id
    _, r2, _ = res2.id
    return (r1 in reg1 and r2 in reg2) or (r1 in reg2 and r2 in reg1)
  return selector

def select_non_none_columns(data: list[list[Optional[int]]]) -> list[list[int]]:
  arr = np.row_stack([np.array(row, dtype=float) for row in data])
  arr = arr[:,~np.any(np.isnan(arr), axis=0)].astype(int)
  return arr.tolist()


def map_pdb_residues(
    row: pd.Series,
    pdb_mappings: dict[tuple[str, str], dict[int, int]],
    region: tuple[int, int],
    from_msa_map: dict[str, dict[int, int]]
  ) -> list[Optional[int]]:
  pdb_map = pdb_mappings[(row["up"], row["pdb"])]
  region_range_in_msa = range(region[0], region[1]+1)
  region_in_up = [
    from_msa_map[row["up"]].get(x)
    for x in region_range_in_msa
  ]
  mapped = [
    pdb_map.get(x) if x else None
    for x in region_in_up
  ]
  return mapped