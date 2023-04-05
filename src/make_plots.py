import os
import pickle
from typing import Optional
import warnings

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from xi_covutils.distances import Distances, calculate_distances
from xi_covutils.pdbmapper import map_align

from make_chimera_plots import (chain_specific_atom_selector,
                                generate_pdb_mapper, get_pdb_file_path,
                                load_cov_data, read_ref_msa)


def load_superimpose_cov_data(folder:str) -> pd.DataFrame :
  infile = os.path.join(
    folder,
    "input",
    "superimpose_pdbs_and_cov.csv"
  )
  df = pd.read_csv(infile)
  return df

def get_color_map():
  colors = [
    [1.0, 1.0, 1.0], # Background
    [1.0, 0.5, 0.5], # Contact in 1 pdb
    [0.5, 1.0, 0.5], # contact in 2 pdbs
    [0.5, 0.5, 1.0], # contact in 3 pdbs
    [.95, .95, .95], # cre/kinase bg
    [0.3, 0.3, 0.3], # Unmapped
    [0.8, 0.6, 0.4], # contact in 1 and cov
    [0.4, 0.8, 0.6], # contact in 2 and cov
    [0.6, 0.4, 0.8], # contact in 3 and cov
    [0.5, 0.5, 0.5], # cov and not contact
    [0.8, 0.0, 0.9], # Cat site
    [0.0, 0.0, 0.0]  # Diagonal
  ]
  my_cmap = LinearSegmentedColormap.from_list(
    "contacts",
    colors,
    N=len(colors)
  )
  return my_cmap

def dstack_product(x, y):
  return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

def add_covdata_to_matrix(
    cov_data,
    max_up
  ):
  df = (
    pd.DataFrame(
      np.hstack(
        (
          dstack_product(np.arange(max_up), np.arange(max_up)),
          np.zeros((max_up**2, 1))
        )
      ),
      columns = ["pos1", "pos2", "score"]
    )
    .assign(
      pos1 = lambda x: x.pos1.astype(int) + 1,
      pos2 = lambda x: x.pos2.astype(int) + 1
    )
    .set_index(["pos1", "pos2"])
  )
  cov = (
    cov_data
      .drop("in_cre_kd", axis = 1)
      .assign(score = 1)
      .pipe(
        lambda df:
          pd.concat(
            (
              df,
              df.rename(
                columns = {"pos1": "pos2", "pos2":"pos1"}
              )
            )
          )
      )
      .reset_index(drop = True)
      .set_index(["pos1", "pos2"])
  )
  df.update(cov)
  df = (
    df
      .reset_index()
      .pivot(
        index = "pos1",
        columns = "pos2",
        values = "score"
      )
  )
  return df.to_numpy()

def create_contact_matrix(
    max_up,
    mappdbs,
    chains,
    seq_mappers,
    pdb_distances,
    cres,
    kinases,
    cov_data,
    cat_sites:Optional[list[list[int]]] = None
  ):
  contact_map = np.zeros(shape = (max_up, max_up))
  background_map = np.zeros(shape = (max_up, max_up))
  for i in range(max_up-1):
    for j in range(i, max_up):
      if i == j:
        background_map[i, j] = 11
        continue
      up_data = zip(mappdbs, chains, seq_mappers, pdb_distances, cres, kinases)
      for mappdb, chain, seq_mapper, dist, cre, kinase in up_data:
        up_pos1 = seq_mapper.get(i+1)
        up_pos2 = seq_mapper.get(j+1)
        if not up_pos1 or not up_pos2:
          # Unmapped
          # background_map[i, j] = 5
          # background_map[j, i] = 5
          continue
        r1 = mappdb.from_seq_to_residue_number(up_pos1)
        r2 = mappdb.from_seq_to_residue_number(up_pos2)
        if not r1 or not r2:
          # background_map[i, j] = 5
          # background_map[j, i] = 5
          continue
        if dist.is_contact(chain, r1, chain, r2):
          contact_map[i, j] += 1
          contact_map[j, i] += 1
      up_data = zip(mappdbs, chains, seq_mappers, pdb_distances, cres, kinases)
      for mappdb, chain, seq_mapper, dist, cre, kinase in up_data:
        cre_range = range(cre[0], cre[1]+1)
        kinase_range = range(kinase[0], kinase[1]+1)
        up_pos1 = seq_mapper.get(i+1)
        up_pos2 = seq_mapper.get(j+1)
        if not up_pos1 or not up_pos2:
          continue
        up1_in_cre = up_pos1 in cre_range
        up1_in_kinase = up_pos1 in kinase_range
        up2_in_cre = up_pos2 in cre_range
        up2_in_kinase = up_pos2 in kinase_range
        if (up1_in_cre and up2_in_kinase) or (up1_in_kinase and up2_in_cre):
          background_map[i, j] = 4
          background_map[j, i] = 4
  if cat_sites:
    for c_sites, seq_mapper in zip(cat_sites, seq_mappers):
      rev_seqmapper = {v:k for k, v in seq_mapper.items()}
      for c_site in c_sites:
        pos = rev_seqmapper.get(c_site)
        for i in range(max_up-1):
          background_map[pos, i] = 10
          background_map[i, pos] = 10
  final_map = background_map
  final_map[contact_map.nonzero()] = contact_map[contact_map.nonzero()]
  df = add_covdata_to_matrix(cov_data, max_up)
  df[contact_map.nonzero()] = (
    df[contact_map.nonzero()]*5
    + contact_map[contact_map.nonzero()]
  )
  df[(contact_map==0).nonzero()] = (
    df[(contact_map==0).nonzero()] * 9
  )
  final_map[df.nonzero()] = df[df.nonzero()]
  return final_map

def load_catsites(folder:str) -> dict[str, list[int]]:
  catsites_file = os.path.join(
    folder,
    "input",
    "cat_sites_uniprot.tsv"
  )
  df = (
    pd.read_csv(
      catsites_file,
      sep="\t",
      header=0
    )
    .groupby("uniprot")
    .apply(
      lambda x: x["pos"].to_list()
    )
    .to_dict()
  )
  return df

def make_superimposed_cov_data_plot(
    data: pd.DataFrame,
    folder:str="",
    include_catsites:bool=False
  ) -> float:
  up_ref = data["up_ref"].iloc[0]
  msa_data = read_ref_msa(up_ref)
  ref_len = len(msa_data[up_ref])
  map_pdbs = [
    generate_pdb_mapper(
      msa_data[u].replace("-", ""),
      p,
      c
    )
    for u, p, c in zip(data["up"], data["pdb"], data["chain"])
  ]
  if include_catsites:
    cat_sites_map = load_catsites(folder)
    cat_sites = [
      cat_sites_map[up]
      for up in data["up"]
    ]
  else:
    cat_sites = []
  # exit()
  seq_mappers = [
    map_align(msa_data[up_ref], msa_data[u])
    for u in data["up"]
  ]
  max_up = max_uniprot_position_with_pdb(
    seq_mappers=seq_mappers,
    mappdbs=map_pdbs,
    ref_up_len=ref_len
  )
  distances = []
  for p, c in zip(data["pdb"], data["chain"]):
    distances.append(
      calc_distances(
        get_pdb_file_path(p),
        c,
        folder
      )
    )
  cres = (
    data[["cre_start", "cre_end"]]
      .apply(tuple, axis=1)
      .to_list()
  )
  kinases = (
    data[["kinase_start", "kinase_end"]]
      .apply(tuple, axis=1)
      .to_list()
  )
  cov_data = load_cov_data(up_ref, "top1")
  contact_map = create_contact_matrix(
    max_up,
    map_pdbs,
    chains = data["chain"].to_list(),
    seq_mappers = seq_mappers,
    pdb_distances = distances,
    cres = cres,
    kinases = kinases,
    cov_data = cov_data,
    cat_sites = cat_sites
  )
  my_cmap = get_color_map()
  fig, axes = plt.subplots(figsize=(40, 40))
  axes.imshow(contact_map, cmap=my_cmap)
  y_ticks, ytick_labels = zip(*[
    (i, i+1)
    for i in range(min(max_up, ref_len))
    if (i+1) % 20 == 0
  ])
  axes.set_yticks(y_ticks)
  axes.set_yticklabels(ytick_labels, fontsize=30)
  x_ticks, xtick_labels = zip(*[
    (i, i+1)
    for i in range(min(max_up, ref_len))
    if (i+1) % 20 == 0
  ])
  axes.set_xticks(x_ticks)
  axes.set_xticklabels(xtick_labels, fontsize=30)
  axes.set_xlabel(f"{up_ref}", fontsize=40)
  axes.set_ylabel(f"{up_ref}", fontsize=40)
  up_pdb_list = " ".join(
    data[["up", "pdb"]]
      .apply(tuple, axis=1)
      .apply(str)
      .to_list()
  )
  axes.set_title(f"MSA ref: {up_ref} - [{up_pdb_list}]", fontsize=45, pad=40)
  fig.tight_layout()
  fig.subplots_adjust(top = 0.96)
  outfile = os.path.join(
    folder,
    "output",
    up_ref,
    f"contact_map_{up_ref}.png"
  )
  fig.savefig(outfile)
  plt.close()
  return 0.0

def max_uniprot_position_with_pdb(
    seq_mappers,
    mappdbs,
    ref_up_len,
  ):
  data = zip(seq_mappers, mappdbs)
  up_pos = set()
  for seq_map, pdb_map in data:
    for p in range(ref_up_len):
      sp = seq_map.get(p+1)
      if not sp:
        continue
      pp = pdb_map.from_seq_to_residue_number(sp)
      if not pp:
        continue
      up_pos.add(p+1)
  max_up = max(up_pos)
  return max_up

def calc_distances(infile, chain, data_folder):
  fn = os.path.basename(infile).replace(".pdb", f"_{chain}.pickle")
  fn = os.path.join(data_folder, "cache", fn)
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  if os.path.exists(fn):
    with open(fn, "rb") as f_in:
      return pickle.load(f_in)
  dist = calculate_distances(
    infile,
    chain_specific_atom_selector(chain)
  )
  dist = [x for x in dist if len(x)==5]
  dist = Distances(dist)
  with open(fn, "wb") as outfile:
    pickle.dump(dist, outfile)
  return dist

def collect_summfiles(data):
  output_folder = os.path.join(data, "output")
  results = []
  for fd, _, fs in os.walk(output_folder):
    for f in fs:
      if "summ" in f and "csv" in f:
        results.append(os.path.join(fd, f))
  return results

def append_inverted_pdbs(df: pd.DataFrame) -> pd.DataFrame:
  return pd.concat(
    [
      df,
      df.rename(
        columns = {"pdb1":"pdb2", "pdb2":"pdb1"}
      )
    ]
  )

def mds_plot_distance(summ_file:str):
  dist = (
    pd
      .read_csv(summ_file)
      .loc[:, ["pdb1", "pdb2", "aln_max"]]
      .pipe(append_inverted_pdbs)
      .pivot(values="aln_max", index="pdb1", columns="pdb2")
      .dropna(how="all", axis=1)
      .dropna(how="all", axis=0)
      .fillna(0)
  )
  if dist.empty:
    return
  mds = MDS(dissimilarity="precomputed", random_state=0)
  transformed = mds.fit_transform(dist)
  if not isinstance(transformed, np.ndarray):
    return
  lim_min = transformed.min()
  lim_max = transformed.max()
  range_abs = abs(lim_max - lim_min)
  spacer = range_abs / 20
  fig, axes = plt.subplots(figsize=(15,15))
  axes.scatter(
    transformed[:, 0],
    transformed[:, 1]
  )
  axes.set_xlim((lim_min-spacer, lim_max+spacer))
  axes.set_ylim(lim_min-spacer, lim_max+spacer)
  for i, pdb_if in enumerate(dist.columns):
    if not isinstance(pdb_if, str):
      continue
    x = transformed[i, 0]
    y = transformed[i, 1]
    if isinstance(x, float) and isinstance(y, float):
      axes.text(x, y, str(pdb_if))
  up = (
    os.path.basename(summ_file)
      .replace("distance_summary_", "")
      .replace(".csv", "")
  )
  axes.set_title(up)
  outfile = summ_file.replace(".csv", "_mds.png")
  fig.savefig(outfile)
  plt.close()

@click.group()
def cli():
  pass

@click.command()
@click.option("--data", default="data", help="Data folder")
def create_cre_cre_mds_plots(data):
  print("Creating MDS plots of CRE-CRE distances")
  summfiles = collect_summfiles(data)
  for sf in summfiles:
    mds_plot_distance(sf)

@click.command()
@click.option("--data", default="data", help="Data folder")
def create_contact_maps(data):
  print("Creating contact maps")
  superimpose_data = load_superimpose_cov_data("data3")
  (
    superimpose_data
      .groupby("up_ref")
      .apply(make_superimposed_cov_data_plot, folder=data)
  )

@click.command()
@click.option("--data", default="data", help="Data folder")
def create_contact_maps_with_catsites(data):
  print("Creating contact maps with catalytic sites")
  superimpose_data = load_superimpose_cov_data("data3")
  (
    superimpose_data
      .groupby("up_ref")
      .apply(
        make_superimposed_cov_data_plot,
        folder=data,
        include_catsites=True
      )
  )

cli.add_command(create_cre_cre_mds_plots)
cli.add_command(create_contact_maps)
cli.add_command(create_contact_maps_with_catsites)

if __name__ == "__main__":
  warnings.simplefilter(
    action='ignore',
    category=FutureWarning
  )
  warnings.simplefilter(
    action='ignore',
    category=PDBConstructionWarning
  )
  cli()
