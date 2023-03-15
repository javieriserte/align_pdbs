#!/usr/bin/env python
"""
App to create Chimera scripts.
"""

#pylint: disable=missing-function-docstring
import itertools
import json
import os
import re
from typing import Any, Callable
import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
from xi_covutils.pdbmapper import map_align, PDBSeqMapper
from xi_covutils.distances import (
    calculate_distances, Distances, DistanceDataSH, DistanceData
)
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from xi_covutils.msa import as_desc_seq_dict, MsaDescSeqDict

os.environ['NO_PROXY'] = '127.0.0.1'

def generate_pdb_mapper(sequence: str, pdb_id:str, chain:str) -> PDBSeqMapper:
  pdb_file = os.path.join(
    "data3",
    "input",
    "pdb",
    f"{pdb_id}.pdb"
  )
  mapper = PDBSeqMapper()
  mapper.align_sequence_to_pdb(sequence, pdb_file, chain)
  return mapper

def get_pdb_file_path(pdb_id:str) -> str:
  f = os.path.join(
    "data3",
    "input",
    "pdb",
    f"{pdb_id}.pdb"
  )
  f = os.path.realpath(f)
  return f

def generate_scripts(series: pd.Series) -> Any:
  ref_up = series.ref
  msa_data = read_ref_msa(ref_up)
  print(ref_up)
  up1 = series.up1
  up2 = series.up2
  chain1 = series.chain1
  chain2 = series.chain2
  pdb1 = series.pdb1
  pdb2 = series.pdb2
  # Los valores de covariacion corresponden a la secuencia
  # de referencia.
  # Hay que mapear las posiciones de la secuencia de referencia
  # a las secuencias con PDB...
  # Y hay que mapear estas posiciones a los numero de residuos anotados
  # en los PDBs.

  # Crea el mapeo de la secuencia de referencia
  # A las secuencias uniprot con PDBs
  # mapr1 = map_align(msa_data[ref_up], msa_data[up1])
  # mapr2 = map_align(msa_data[ref_up], msa_data[up2])

  pdb_file1 = get_pdb_file_path(pdb1)
  pdb_file2 = get_pdb_file_path(pdb2)

  # Crea el mapeo de las secuencias uniprot a las posicines dentro del PDB.
  mappdb1 = generate_pdb_mapper(
    msa_data[up1].replace("-", ""),
    pdb1,
    chain1
  )
  mappdb2 = generate_pdb_mapper(
    msa_data[up2].replace("-", ""),
    pdb2,
    chain2
  )
  kd1 = series.kd1
  kd1_res = [
    mappdb1.from_seq_to_residue_number(x) for x in range(kd1[0], kd1[1]+1)
  ]
  kd1_res = sorted([x for x in kd1_res if x])
  if not kd1_res:
    print("Not KD residues")
    return
  kd2 = series.kd2
  kd2_res = [
    mappdb2.from_seq_to_residue_number(x) for x in range(kd2[0], kd2[1]+1)
  ]
  kd2_res = sorted([x for x in kd2_res if x])
  if not kd2_res:
    print("Not KD residues")
    return
  kd1_atomspec = f"#0:{kd1_res[0]}-{kd1_res[-1]}.{chain1}"
  kd2_atomspec = f"#1:{kd2_res[0]}-{kd2_res[-1]}.{chain2}"
  cre1 = series.cre1
  cre1_res = [
    mappdb1.from_seq_to_residue_number(x) for x in range(cre1[0], cre1[1]+1)
  ]
  cre1_res = sorted([x for x in cre1_res if x])
  if not cre1_res:
    print("Not CRE residues")
    return
  cre2 = series.cre2
  cre2_res = [
    mappdb2.from_seq_to_residue_number(x) for x in range(cre2[0], cre2[1]+1)
  ]
  cre2_res = sorted([x for x in cre2_res if x])
  if not cre2_res:
    print("Not KD residues")
    return
  cre1_atomspec = f"#0:{cre1_res[0]}-{cre1_res[-1]}.{chain1}"
  cre2_atomspec = f"#1:{cre2_res[0]}-{cre2_res[-1]}.{chain2}"
  cmds = [
    chimera_close_session(),
    chimera_background(["solid", "white"]),
    chimera_open(pdb_file1),
    chimera_open(pdb_file2),
    chimera_hide_ribbon(),
    chimera_hide_display(),
    chimera_color_ribbon(f"#0:.{chain1}", "lightgray"),
    chimera_color_ribbon(f"#1:.{chain2}", "lightgray"),
    chimera_color_ribbon(kd1_atomspec, "hotpink"),
    chimera_color_ribbon(kd2_atomspec, "steelblue"),
    chimera_color_ribbon(cre1_atomspec, "orangered"),
    chimera_color_ribbon(cre2_atomspec, "deepskyblue"),
    chimera_ribbon(f"#0:.{chain1}"),
    chimera_ribbon(f"#1:.{chain2}"),
    chimera_mmaker(kd1_atomspec, kd2_atomspec)
  ]

  cov_file = get_cov_file(ref_up, "top1")
  if os.path.exists(cov_file):
    cov_data = (
      load_cov_data(ref_up, "top1")
        .assign(
          mapped1 = lambda x:
            x.pos1.apply(mappdb1.from_seq_to_residue_number)
        )
        .assign(
          mapped2 = lambda x:
            x.pos2.apply(mappdb1.from_seq_to_residue_number)
        )
        .dropna()
        .assign(
          atomspec1 =
            lambda x :
              x.apply(
                lambda y: (
                  f"#0:{int(y.mapped1)}.{chain1}@CA"
                  f"#0:{int(y.mapped2)}.{chain2}@CA"
                ),
                axis = 1
              )
        )
        .assign(
          atomspec2 =
            lambda x :
              x.apply(
                lambda y: (
                  f"#1:{int(y.mapped1)}.{chain1}@CA"
                  f"#1:{int(y.mapped2)}.{chain2}@CA"
                ),
                axis = 1
              )
        )
        .assign(
          cmd1 = lambda y:
            y.apply(
              lambda x: chimera_shape_tube(x.atomspec1, "red", 0.1),
              axis = 1
            )
        )
        .assign(
          cmd2 = lambda y:
            y.apply(
              lambda x: chimera_shape_tube(x.atomspec2, "blue", 0.1),
              axis = 1
            )
        )
    )
    tube_cmds = [
      # x for x in itertools.chain(cov_data.cmd1, cov_data.cmd2)
      x for x in itertools.chain(cov_data.cmd1)
    ]
    cmds = cmds + tube_cmds

  export_cmds(ref_up, cmds)

  # for cmd in cmds:
  #   result = chimera_command(cmd, 39205)
  #   print(result)

def export_cmds(ref_up:str, cmds:list[str]):
  cmd_file = os.path.join(
    "data3",
    "output",
    "chimera_scripts",
    f"{ref_up}.cmd"
  )
  os.makedirs(os.path.dirname(cmd_file), exist_ok=True)
  with open(cmd_file, 'w', encoding='utf8') as f_out:
    for cmd in cmds:
      f_out.write(f"{cmd}\n")

def chimera_shape_tube(
    atomspec:str,
    color:str,
    radius:float
  ) -> str:
  return f"shape tube {atomspec} r {radius} color {color}"

def chimera_color_ribbon(atomspec:str, color: str) -> str:
  return f"color {color},r {atomspec}"

def chimera_hide_display():
  return "~display"

def chimera_ribbon(atomspec: str) -> str:
  return f"ribbon {atomspec}"

def chimera_hide_ribbon():
  return "~ribbon"

def chimera_mmaker(atomspec1: str, atomspec2: str) -> str:
  return f"mmaker {atomspec1} {atomspec2}"

def chimera_background(args: list[str]) -> str:
  return f"background {' '.join(args)}"

def chimera_close_session():
  return "close session"

def chimera_command(cmd: str, port: int) -> str:
  print(cmd)
  url = f"http://127.0.0.1:{port}/run"
  response = requests.get(
    url,
    params = {"command": cmd},
    timeout=200
  )
  return response.text

def chimera_open(file) -> str:
  return f"open {file}"


def get_cov_file(up_acc: str, top: str) -> str:
  cov_file = os.path.join(
    "data3",
    "input",
    "cov",
    f"covariation_{top}_{up_acc}.csv"
  )
  return cov_file

def load_cov_data(ref_up:str, top: str) -> pd.DataFrame:
  cov_file = get_cov_file(ref_up, top)
  data = (
    pd
      .read_csv(cov_file)
      .set_axis(
        ["pos1", "pos2", "score", "in_cre_kd"],
        axis=1
      )
      .query(
        "abs(pos1-pos2)>5 and in_cre_kd"
      )
  )
  return data

def read_ref_msa(un_acc: str) -> MsaDescSeqDict:
  msa_file = os.path.join(
    "data3",
    "input",
    f"{un_acc}_clustalO_fullidentity_60.fasta.aln"
  )
  return as_desc_seq_dict(msa_file)

def get_region_data_file() -> str:
  data_file = os.path.join(
    "data3",
    "input",
    "df_ups_w_diff.csv"
  )
  return data_file

def read_region_data() -> pd.DataFrame:
  def _str_to_list(series: pd.Series) -> pd.Series:
    return (
      series
        .fillna('[]')
        .apply(json.loads)
    )
  data_file = get_region_data_file()
  data = (
    pd.read_csv(
      data_file,
      index_col=0,
    )
    .assign(phos_sites1 = lambda x: _str_to_list(x.phos_sites1))
    .assign(phos_sites2 = lambda x: _str_to_list(x.phos_sites2))
    .assign(kd1 = lambda x: _str_to_list(x.kd1))
    .assign(kd2 = lambda x: _str_to_list(x.kd2))
    .assign(cre1 = lambda x: _str_to_list(x.cre1))
    .assign(cre2 = lambda x: _str_to_list(x.cre2))
    .assign(cov_file = lambda x: x.ref.apply(get_cov_file, top="top01"))
    .assign(has_cov_data =
      lambda x: x.ref.apply(lambda y : os.path.exists(get_cov_file(y, "top01")))
    )
    # .query("has_cov_data")
    # .iloc[[16], :]
  )

  return data

def get_uniprot_order_from_output():
  output_folder = os.path.join(
    "data3",
    "output"
  )
  results = []
  pattern = re.compile(r"aligned_full_(.+)_(.+)_(.+)_(.+).pdb")
  for fd, _, fs in os.walk(output_folder):
    ref_up = fd.replace("data3/output/", "")
    for f in fs:
      if m := re.match(pattern, f):
        results.append(
          [ref_up, m.group(1), m.group(2), m.group(3), m.group(4)]
        )
  order = pd.DataFrame(
    results,
    columns = ["ref_up", "up1", "up2", "pdb1", "pdb2"]
  )
  return order

def reorganize_region_data():
  data_file = get_region_data_file()
  data = (
    pd.read_csv(
      data_file,
      index_col=0
    )
    .loc[:, ["ref", "up1", "up2", "pdb1", "pdb2"]]
    .set_axis(["ref_up", "up1", "up2", "pdb1", "pdb2"], axis=1)
  )

  data2 = (
    data.set_axis(
      ["ref_up", "up2", "up1", "pdb2", "pdb1"],
      axis =1
    )
    .loc[:,["ref_up", "up1", "up2", "pdb1", "pdb2"]]
  )
  order = get_uniprot_order_from_output()
  data_v1 = data.merge(order)
  data_v2 = data2.merge(order)

  print(data_v1)
  print(data_v2)

def generate_contact_map(series: pd.Series) -> Any:
  ref_up = series.ref
  msa_data = read_ref_msa(ref_up)
  up1 = series.up1
  up2 = series.up2
  chain1 = series.chain1
  chain2 = series.chain2
  pdb1 = series.pdb1
  pdb2 = series.pdb2
  # Los valores de covariacion corresponden a la secuencia
  # de referencia.
  # Hay que mapear las posiciones de la secuencia de referencia
  # a las secuencias con PDB...
  # Y hay que mapear estas posiciones a los numero de residuos anotados
  # en los PDBs.

  # Crea el mapeo de la secuencia de referencia
  # A las secuencias uniprot con PDBs
  # mapr1 = map_align(msa_data[ref_up], msa_data[up1])
  # mapr2 = map_align(msa_data[ref_up], msa_data[up2])

  pdb_file1 = get_pdb_file_path(pdb1)
  pdb_file2 = get_pdb_file_path(pdb2)

  # Crea el mapeo de las secuencias uniprot a las posicines dentro del PDB.
  mappdb1 = generate_pdb_mapper(
    msa_data[up1].replace("-", ""),
    pdb1,
    chain1
  )
  mappdb2 = generate_pdb_mapper(
    msa_data[up2].replace("-", ""),
    pdb2,
    chain2
  )
  kd1 = series.kd1
  kd1_res = [
    mappdb1.from_seq_to_residue_number(x) for x in range(kd1[0], kd1[1]+1)
  ]
  kd1_res = sorted([x for x in kd1_res if x])
  if not kd1_res:
    print("Not KD residues")
    return
  kd2 = series.kd2
  kd2_res = [
    mappdb2.from_seq_to_residue_number(x) for x in range(kd2[0], kd2[1]+1)
  ]
  kd2_res = sorted([x for x in kd2_res if x])
  if not kd2_res:
    print("Not KD residues")
    return
  cre1 = series.cre1
  cre1_res = [
    mappdb1.from_seq_to_residue_number(x) for x in range(cre1[0], cre1[1]+1)
  ]
  cre1_res = sorted([x for x in cre1_res if x])
  if not cre1_res:
    print("Not CRE residues")
    return
  cre2 = series.cre2
  cre2_res = [
    mappdb2.from_seq_to_residue_number(x) for x in range(cre2[0], cre2[1]+1)
  ]
  cre2_res = sorted([x for x in cre2_res if x])
  if not cre2_res:
    print("Not KD residues")
    return
  pdb1_distances = calculate_distances(
    pdb_source=pdb_file1,
    atom_selector=chain_specific_atom_selector(chain1)
  )
  pdb1_distances = [x for x in pdb1_distances if len(x) == 5]
  pdb1d = Distances(pdb1_distances)
  pdb2_distances = calculate_distances(
    pdb_source=pdb_file2,
    atom_selector=chain_specific_atom_selector(chain2)
  )
  pdb2_distances = [x for x in pdb2_distances if len(x) == 5]
  pdb2d = Distances(pdb2_distances)
  all_residues1 = collect_residues(pdb1_distances)
  all_residues2 = collect_residues(pdb2_distances)
  contact_map = np.zeros(
    shape = (
      len(all_residues1), len(all_residues2)
    )
  )
  for i, ri in enumerate(all_residues1):
    for j, rj in enumerate(all_residues2):
      uri = mappdb1.from_residue_number_to_seq(ri)
      urj = mappdb2.from_residue_number_to_seq(rj)
      if not uri or not urj:
        continue
      if uri == urj:
        contact_map[i, j] = 3
      if uri > urj:
        contact_map[i, j] = 1 if pdb1d.is_contact(chain1, ri, chain2, rj) else 0
      if uri < urj:
        contact_map[i, j] = 2 if pdb2d.is_contact(chain1, ri, chain2, rj) else 0
  fig, axes = plt.subplots()
  axes.imshow(contact_map, cmap="set1")
  fig.show()

def collect_residues(dist_data: DistanceData) -> list[int]:
  resset = {
    int(resnum)
    for distelem in dist_data
    for resnum in (distelem[1], distelem[3])
  }
  return sorted(resset)

def chain_specific_atom_selector(chain: str) -> Callable[[Atom, Atom], bool]:
  def _selector(atom1: Atom, atom2: Atom) -> bool:
    res1 = atom1.parent
    res2 = atom2.parent
    if not isinstance(res1, Residue) or not isinstance(res2, Residue):
      return False
    chain1 = res1.parent
    chain2 = res2.parent
    if not isinstance(chain1, Chain) or not isinstance(chain2, Chain):
      return False
    return (
      chain1.id == chain and chain2.id == chain
    )
  return _selector

@click.command()
def create_chimera_scripts():
  """
  Create Chimera Scripts commands.
  """
  print("We are busy create some Chimera scripts for you, please wait....\n\n")
  data:pd.DataFrame = read_region_data()
  (
    data
      .apply(generate_scripts, axis=1)
  )

@click.command()
def create_contact_maps():
  """
  Create contact Map commands.
  """
  print("We are create some random contact maps for you, please wait... \n\n")
  data:pd.DataFrame = read_region_data()
  (
    data
      .query("has_cov_data")
      .iloc[[0], :]
      .apply(generate_contact_map, axis=1)
  )

@click.group()
def cli():
  """
  Main App command
  """
  pass

cli.add_command(create_chimera_scripts)
cli.add_command(create_contact_maps)

if __name__ == "__main__":
  cli()
