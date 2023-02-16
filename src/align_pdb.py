import json
import os
from typing import Dict, List, Tuple
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from copy import deepcopy
import pandas as pd
from xi_covutils.distances import calculate_distances

def load_domains():
  domains_file = os.path.join(
    "data",
    "input",
    "domains.json"
  )
  with open(domains_file, "r", encoding="utf8") as f_in:
    domains = json.load(f_in)
  return domains

def load_aln():
  aln_file = os.path.join(
    "data",
    "input",
    "aln.fa"
  )
  records = {
    r.id : str(r.seq)
    for r in SeqIO.parse(aln_file, "fasta")
  }
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
  ) -> Tuple[List[int], List[int]]:
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
  mapped_positions = [
    (p1, p2)
    for p1, p2 in mapped_positions
    if p1 and p2
  ]
  mapped_positions = list(zip(*mapped_positions))
  return mapped_positions

def load_pdb_mappings(up_map):
  result = {}
  for k, v in up_map.items():
    infile = os.path.join(
      "data",
      "input",
      f"{v[0]}.map"
    )
    with open(infile, "r", encoding="utf-8") as f_in:
      data = json.load(f_in)
      data = {
        int(p1): int(p2)
        for p1, p2 in data.items()
      }
    result[k] = data
  return result

def load_up_maps():
  mapping_file = os.path.join(
    "data",
    "input",
    "up_pdb_map.json"
  )
  with open(mapping_file, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
  return data

def retain_region(struct, chain, region):
  for c in struct[0].get_chains():
    if c.id != chain:
      c.parent.detach_child(c.id)
    else:
      residues = list(c.get_residues())
      for res in residues:
        c.detach_child(res.id)
        if res.id[1] in region:
          c.add(res)

def export_aligned_full_pdbs(up1, up2, up_map, struc1, struc2):
  st = Structure("Aligned")
  st.add(struc1[0])
  m2 = deepcopy(struc2[0])
  m2.id=1
  st.add(m2)
  io = PDBIO()
  io.set_structure(st)
  outfile = os.path.join(
    "data",
    "output",
    f"aligned_full_{up1}_{up2}"
    f"_{up_map[up1][0]}"
    f"_{up_map[up2][0]}"
    ".pdb"
  )
  io.save(outfile)

def main():
  # Carga datos de los dominios.
  domains = load_domains()

  # Lee los uniprot accession de las proteinas a alinear.
  up1, up2 = list(domains.keys())

  # Carga el alineamiento entre las dos secuencias.
  aln = load_aln()

  # Carga el mapeo de uniprot accession a PDB Id y cadena.
  up_chain_map = load_up_maps()

  # Carga el mapeo de las posiciones de las secuencias completas
  # segun Uniprot y los numeros de residuos en el PDB.
  up_pdb_pos_map = load_pdb_mappings(up_chain_map)

  # Extrae las posiciones PDBs de los dominios kinasas.
  # Se buscan posiciones que el alineamiento no tengan gaps y
  # que esten dentro de la region del dominio kinasa en
  # ambas proteinas.
  pdbpos1, pdbpos2 = region_in_pdb(
    aln["Q92918"],
    aln["Q8IVH8"],
    domains["Q92918"]["kinase"],
    domains["Q8IVH8"]["kinase"],
    up_pdb_pos_map["Q92918"],
    up_pdb_pos_map["Q8IVH8"]
  )

  # Lee los archivos PDBs
  struc1, struc2 = [
    read_pdb(x, up_chain_map) for x in (up1, up2)
  ]

  # Extrae los atomos comparables de los dominios Kinasas
  kinase1 = [
    struc1[0][up_chain_map[up1][1]][r]["CA"]
    for r in pdbpos1
  ]
  kinase2 = [
    struc2[0][up_chain_map[up2][1]][r]["CA"]
    for r in pdbpos2
  ]

  # Alinear los atomos y obtener la matriz de rotacion
  # set_atoms, genera las matrices de translacion y rotacion.
  # El primer conjunto de atomos esta fijo y el segundo es el movil.
  sup = Superimposer()
  sup.set_atoms(kinase1, kinase2)

  # Aplica la rotacion a la segunda estructura completa.
  sup.apply(struc2)

  # Exportar full PDB aligned
  # Las dos estructuras se guardan en el mismo archivo pdb como dos
  # modelos diferentes.
  export_aligned_full_pdbs(up1, up2, up_chain_map, struc1, struc2)

  # Extrae los números de residuos de los CRE
  # de las dos estructuras.
  pdbpos_cre1, pdbpos_cre2 = region_in_pdb(
    aln["Q92918"],
    aln["Q8IVH8"],
    domains["Q92918"]["cre"],
    domains["Q8IVH8"]["cre"],
    up_pdb_pos_map["Q92918"],
    up_pdb_pos_map["Q8IVH8"]
  )

  # Elimina todas las cadenas y residuos de las estructuras
  # excepto aquellos que pertenecen al CRE.
  retain_region(struc1, up_chain_map[up1][1], pdbpos_cre1)
  retain_region(struc2, up_chain_map[up2][1], pdbpos_cre2)

  # Mergear los dos cre en una sola estructura
  # La cadena de la primer estructura pasa a ser A
  # y la cadena de la segunda pasa a ser B.
  struc1[0][up_chain_map[up1][1]].id = "A"
  struc2[0][up_chain_map[up2][1]].id = "B"
  struc1[0].add(struc2[0]["B"])

  # Exportar PDB de los CRE
  # Los dos CRE estan en el mismo PDB como dos cadenas
  # diferentes.
  export_aligned_cre_pdb(up1, up2, up_chain_map, struc1)

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
  export_distances(up1, up2, up_chain_map, dist_df)

def export_distances(up1, up2, up_chain_map, dist_df):
  dist_file = os.path.join(
    "data",
    "output",
    f"distances_{up1}_{up2}_"
    f"{up_chain_map[up1][0]}_"
    f"{up_chain_map[up2][0]}"
    ".csv"
  )
  (
    dist_df
      .loc[:, "distance"]
      .reset_index(drop=True)
      .to_csv(dist_file, index=False, header = False)
  )


def read_pdb(up, up_chain_map):
  pdb_file = (
    os.path.join(
      "data",
      "input",
      f"{up_chain_map[up][0]}.pdb",
    )
  )
  struc = PDBParser().get_structure(id=up, file=pdb_file)
  return struc

def export_aligned_cre_pdb(up1, up2, up_chain_map, struct):
    io=PDBIO()
    io.set_structure(struct)
    outfile = os.path.join(
      "data",
      "output",
      f"aligned_cre_{up1}_{up2}"
      f"_{up_chain_map[up1][0]}"
      f"_{up_chain_map[up2][0]}"
      ".pdb"
    )
    io.save(outfile)

if __name__ == '__main__':
  main()