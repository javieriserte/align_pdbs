import shutil
import json
import logging
import os
from copy import deepcopy
from itertools import combinations
from logging import WARNING
from typing import Dict, List, Tuple
from numpy import sqrt

from tqdm import tqdm
import click
import pandas as pd
from Bio import SeqIO
from Bio.PDB import Superimposer, Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from xi_covutils.distances import calculate_distances
from xi_covutils.pdbmapper import align_pdb_to_sequence


def load_domains(folder:str):
  domains_file = os.path.join(
    folder,
    "input",
    "domains.json"
  )
  with open(domains_file, "r", encoding="utf8") as f_in:
    domains = json.load(f_in)
  print(f"- Cargando dominios:")
  print(f"  - Número de dominios: {len(domains)}")
  return domains

def load_aln(folder:str):
  aln_file = os.path.join(
    folder,
    "input",
    "aln.fa"
  )
  records = {
    r.id : str(r.seq)
    for r in SeqIO.parse(aln_file, "fasta")
  }
  print(f"- Cargando alineamiento: {aln_file}")
  print(f"  - Número de secuencias: {len(records)}")
  return records

def enumerate_in_aln(seq_aln:str) -> List[Tuple[int, str]]:
  result = []
  pos = 1
  for s in seq_aln:
    if s == "-":
      result.append((0, s))
    else:
      result.append((pos, s))
      pos += 1
  return result

def region_in_pdb(
    seq_aln1: str,
    seq_aln2: str,
    reg1: Tuple[int, int],
    reg2: Tuple[int, int],
    map1: Dict[int, int],
    map2: Dict[int, int]
  ) -> Tuple[Tuple[int], Tuple[int]]:
  uniprot_positions = zip(
    enumerate_in_aln(seq_aln1),
    enumerate_in_aln(seq_aln2),
  )
  wanted_uniprot_positions = [
    (p1, p2)
    for ((p1, s1), (p2, s2)) in uniprot_positions
    if p1 >= reg1[0] and p1 <= reg1[1]
    if p2 >= reg2[0] and p2 <= reg2[1]
    if s1 != "-" and s2 != "-"
  ]
  mapped_positions = [
    (
      map1.get(p1, None),
      map2.get(p2, None)
    )
    for p1, p2 in wanted_uniprot_positions
  ]
  mapped_positions_2: List[Tuple[int, int]] = [
    (p1, p2)
    for p1, p2 in mapped_positions
    if p1 and p2
  ]
  mapped_positions_3: Tuple[Tuple[int], Tuple[int]] = tuple(zip(*mapped_positions_2))
  return mapped_positions_3

def load_pdb_mappings(up_map, folder):
  result = {}
  for k, v in up_map.items():
    for pdb, _ in v:
      infile = os.path.join(
        folder,
        "intermediate",
        "pdb_mapping",
        f"{k}_{pdb}.map"
      )
      if not os.path.exists(infile):
        continue
      with open(infile, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
        data = {
          int(p1): int(p2)
          for p1, p2 in data.items()
        }
      result[(k, pdb)] = data
  print(f"- Cargando Mapeo de posiciones de Uniprot a PDB:")
  print(f"  - Mapeos cargados: {len(result)}")
  return result

def load_up_maps(folder:str):
  mapping_file = os.path.join(
    folder,
    "input",
    "up_pdb_map.json"
  )
  with open(mapping_file, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
  print(f"- Cargando Mapeo de identificadores de Uniprot a PDB/Cadena")
  print(f"  - Número de Uniprots: {len(data)}")
  return data

def retain_region(struct, chain, region):
  chains_to_detach = []
  for c in struct[0].get_chains():
    if c.id != chain:
      chains_to_detach.append(c.id)
    else:
      residues = list(c.get_residues())
      for res in residues:
        c.detach_child(res.id)
        if res.id[1] in region:
          c.add(res)
  for c in chains_to_detach:
    struct[0].detach_child(c)

def export_aligned_full_pdbs(folder, job, up1, up2, pdb1, pdb2, struc1, struc2):
  st = Structure("Aligned")
  m1 = struc1[0]
  m1.id = 0
  st.add(m1)
  m2 = deepcopy(struc2[0])
  m2.id=1
  st.add(m2)
  io = PDBIO()
  io.set_structure(st)
  outfile = os.path.join(
    folder,
    "output",
    job,
    f"aligned_full_{up1}_{up2}"
    f"_{pdb1}"
    f"_{pdb2}"
    ".pdb"
  )
  os.makedirs(os.path.dirname(outfile), exist_ok=True)
  io.save(outfile)

def read_sequences(folder:str):
  seq_file = os.path.join(
    folder,
    "input",
    "sequences.fa"
  )
  records = {
    r.id : str(r.seq)
    for r in SeqIO.parse(seq_file, "fasta")
  }
  return records

def create_up_to_pdb_maps(aln, up_pdb_map, folder):
  outfolder = os.path.join(
    folder,
    "intermediate",
    "pdb_mapping"
  )
  shutil.rmtree(outfolder)
  os.makedirs(outfolder)
  for up, sequence in aln.items():
    for pdb_id, chain in up_pdb_map.get(up, []):
      pdb_file = os.path.join(
        folder,
        "input",
        "pdb",
        f"{pdb_id}.pdb"
      )
      if not os.path.exists(pdb_file):
        continue
      sequence = sequence.replace("-", "")
      mapped = align_pdb_to_sequence(pdb_file, chain, sequence)
      mapped = {v:k for k,v in mapped.items()}
      outfile = os.path.join(outfolder, f"{up}_{pdb_id}.map")
      with open(outfile, "w", encoding="utf8") as f_out:
        json.dump(mapped, f_out, indent=2)

def get_cre_distance_two_kinases(
    up1, pdb1, chain1, seq1,
    up2, pdb2, chain2, seq2,
    domains,
    up_pdb_pos_map,
    folder,
    job
  ):
  kinase1_res = domains[up1]["kinase"]
  kinase2_res = domains[up2]["kinase"]
  cre1_res = domains[up1]["cre"]
  cre2_res = domains[up2]["cre"]
  pdb_map1 = up_pdb_pos_map[(up1, pdb1)]
  pdb_map2 = up_pdb_pos_map[(up2, pdb2)]
  # Extrae las posiciones PDBs de los dominios kinasas.
  # Se buscan posiciones que el alineamiento no tengan gaps y
  # que esten dentro de la region del dominio kinasa en
  # ambas proteinas.
  pdbpos1, pdbpos2 = region_in_pdb(
    seq1,
    seq2,
    kinase1_res,
    kinase2_res,
    pdb_map1,
    pdb_map2
  )
  # Lee los archivos PDBs
  struc1 = read_pdb(pdb1, up1, folder)
  struc2 = read_pdb(pdb2, up2, folder)
  # Extrae los atomos comparables de los dominios Kinasas
  kinase1 = [
    struc1[0][chain1][r]["CA"]
    for r in pdbpos1
  ]
  kinase2 = [
    struc2[0][chain2][r]["CA"]
    for r in pdbpos2
  ]
  # Alinear los atomos y obtener la matriz de rotacion
  # set_atoms, genera las matrices de translacion y rotacion.
  # El primer conjunto de atomos esta fijo y el segundo es el movil.
  sup = Superimposer()
  sup.set_atoms(kinase1, kinase2)
  kin_rmsd = sup.rms
  # Aplica la rotacion a la segunda estructura completa.
  sup.apply(struc2)
  # Exportar full PDB aligned
  # Las dos estructuras se guardan en el mismo archivo pdb como dos
  # modelos diferentes.
  export_aligned_full_pdbs(folder, job, up1, up2, pdb1, pdb2, struc1, struc2)
  # Extrae los números de residuos de los CRE
  # de las dos estructuras.
  reg_in_pdb = region_in_pdb(
    seq1,
    seq2,
    cre1_res,
    cre2_res,
    up_pdb_pos_map[(up1, pdb1)],
    up_pdb_pos_map[(up2, pdb2)]
  )
  if not reg_in_pdb:
    return (None, None, None, None, None)
  pdbpos_cre1, pdbpos_cre2 = reg_in_pdb
  # Elimina todas las cadenas y residuos de las estructuras
  # excepto aquellos que pertenecen al CRE.
  retain_region(struc1, chain1, pdbpos_cre1)
  retain_region(struc2, chain2, pdbpos_cre2)
  # Mergear los dos cre en una sola estructura
  # La cadena de la primer estructura pasa a ser A
  # y la cadena de la segunda pasa a ser B.
  struc1[0][chain1].id = "A"
  struc2[0][chain2].id = "B"
  struc1[0].add(struc2[0]["B"])
  # Exportar PDB de los CRE
  # Los dos CRE estan en el mismo PDB como dos cadenas
  # diferentes.
  export_aligned_cre_pdb(up1, up2, pdb1, pdb2, struc1, folder, job)
  # Calcula las distancias
  dist_data = calculate_distances(struc1)
  # Selecciona las distancias entre los dos CRE
  dist_df = (
    pd.DataFrame(
      dist_data,
      columns = ["chain1", "pos1", "chain2", "pos2", "distance"]
    )
    .query("chain1!=chain2")
  )
  # Exporta las distancias entre los CRE a un CSV
  export_distances(up1, pdb1, up2, pdb2, dist_df, folder, job)
  min_dist = dist_df.distance.min()
  mean_dist = dist_df.distance.mean()
  df_cre = pd.DataFrame(
    {
      "pos1": pdbpos_cre1,
      "pos2": pdbpos_cre2
    }
  )
  aligned_mean = (
    dist_df
      .merge(df_cre, on=["pos1", "pos2"])
      .distance
      .mean()
  )
  rmsd = sqrt(
    dist_df
      .merge(df_cre, on=["pos1", "pos2"])
      .distance
      .pow(2)
      .mean()
  )
  return min_dist, mean_dist, aligned_mean, rmsd, kin_rmsd

@click.command()
@click.argument('folder', type=click.Path(exists=True), default="data")
@click.argument('job', type=click.STRING, default="default_job")
def align_kinases(folder, job):
  print("Align Kinases PDBs and Calculate CRE distances\n")
  # Carga el alineamiento entre las dos secuencias.
  aln = load_aln(folder)
  # Carga el mapeo de uniprot accession a PDB Id y cadena.
  up_chain_map = load_up_maps(folder)
  # Obtiene una lista de datos de cada combinacion de Uniprot/PDB
  up_pdbs = [
    (up, pdb_id, chain, seq)
    for up, seq in aln.items()
    for pdb_id, chain in up_chain_map.get(up, [])
  ]
  if len(up_pdbs) <= 1:
    print("Not enough pdbs to compare.")
    return
  pairs_to_compare = combinations(up_pdbs, 2)
  # Carga datos de los dominios.
  domains = load_domains(folder)
  # Carga el mapeo de las posiciones de las secuencias completas
  # segun Uniprot y los numeros de residuos en el PDB.
  up_pdb_pos_map = load_pdb_mappings(up_chain_map, folder)
  # Alinea las kinasas y calcula las distancias del CRE en todos los pares
  if not up_pdb_pos_map:
    print("Este alineamiento parece no tener secuencias con PDBs.")
    return
  distances = []
  print("- Alineando y calculando distancias:")
  for elem1, elem2 in (pbar:= tqdm(pairs_to_compare)):
    up1, pdb1, chain1, seq1 = elem1
    up2, pdb2, chain2, seq2 = elem2
    pbar.set_description(f"  - {up1}-{pdb1}:{chain1} vs. {up2}-{pdb2}:{chain2}")
    mind, meand, aln_mean, rmsd, kin_rmsd = get_cre_distance_two_kinases(
      up1, pdb1, chain1, seq1,
      up2, pdb2, chain2, seq2,
      domains,
      up_pdb_pos_map,
      folder,
      job
    )
    distances.append(
      [
        up1, pdb1, chain1,
        up2, pdb2, chain2,
        mind, meand, aln_mean, rmsd, kin_rmsd
      ]
    )
  # Expotar distance summary
  export_distance_summary(distances, folder, job)

def export_distance_summary(distances, folder, job_name):
  dist_file = os.path.join(
    folder,
    "output",
    job_name,
    f"distance_summary_{job_name}.csv"
  )
  os.makedirs(os.path.dirname(dist_file), exist_ok=True)
  pd.DataFrame(
    distances,
    columns = [
      "up1", "pdb1", "chain1",
      "up2", "pdb2", "chain2",
      "min_distance",
      "mean_distance",
      "aligned_mean_distance",
      "rmsd",
      "rmsd kinases"
    ]
  ).to_csv(
    dist_file,
    index = False
  )
  print(f"- Guardando resumen de distancias: {dist_file}")


def export_distances(up1, pdb1, up2, pdb2, dist_df, folder, job):
  dist_file = os.path.join(
    folder,
    "output",
    job,
    f"distances_{up1}_{up2}_"
    f"{pdb1}_"
    f"{pdb2}"
    ".csv"
  )
  (
    dist_df
      .reset_index(drop=True)
      .to_csv(dist_file, index=False, header = False)
  )

def read_pdb(pdb, up, folder):
  def _fix_alternative_use_residues(struct):
    res_list = Selection.unfold_entities(struc, "R")
    if not res_list:
      raise ValueError("Not PDB")
    for r in res_list:
      if not isinstance(r, Residue):
        continue
      _, b, _ = r.id
      r.id = (' ', b, ' ')
  def _remove_non_wanted_models(struct):
    models = struct.get_models()
    wanted_model = models[0].id
    models_to_remove = [
      m
      for m in models
      if m.id != wanted_model.id
    ]
    for m in models_to_remove:
      m.parent.detach_child(m.id)
  pdb_file = (
    os.path.join(
      folder,
      "input",
      "pdb",
      f"{pdb}.pdb",
    )
  )
  struc = PDBParser().get_structure(id=up, file=pdb_file)
  _fix_alternative_use_residues(struc)
  _remove_non_wanted_models(struc)
  return struc

def export_aligned_cre_pdb(up1, up2, pdb1, pdb2, struct, folder, job):
  io=PDBIO()
  io.set_structure(struct)
  outfile = os.path.join(
    folder,
    "output",
    job,
    f"aligned_cre_{up1}_{up2}"
    f"_{pdb1}"
    f"_{pdb2}"
    ".pdb"
  )
  io.save(outfile)

@click.group()
def cli():
  print("######### Aligner App ##########\n")

@click.command()
@click.argument('folder', type=click.Path(exists=True), default="data")
def create_pdb_mappings(folder):
  print("Generating Uniprot to PDB Mappings\n")
  # Carga el alineamiento entre las dos secuencias.
  aln = load_aln(folder)
  # Carga el mapeo de uniprot accession a PDB Id y cadena.
  up_chain_map = load_up_maps(folder)
  # Crea los mapeos de las secuencias de Uniprot a los PDB.
  create_up_to_pdb_maps(aln, up_chain_map, folder)

cli.add_command(create_pdb_mappings)
cli.add_command(align_kinases)

def config_logging():
  logging.basicConfig(
    filename="log",
    level=logging.WARNING
  )
  logging.captureWarnings(True)

if __name__ == '__main__':
  config_logging()
  cli()