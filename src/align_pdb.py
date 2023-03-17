import shutil
import json
import logging
import os
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Tuple
from numpy import sqrt

import click
from crecredist import create_up_to_pdb_maps, export_distance_summary, get_cre_distance_two_kinases, load_aln, load_domains, load_pdb_mappings, load_up_maps
from kincredist import calc_kin_cre_distances

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
  pairs_to_compare = list(combinations(up_pdbs, 2))
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
  for i, (elem1, elem2) in enumerate(pairs_to_compare):
    up1, pdb1, chain1, seq1 = elem1
    up2, pdb2, chain2, seq2 = elem2
    # pbar.set_description(f"  - {up1}-{pdb1}:{chain1} vs. {up2}-{pdb2}:{chain2}")
    print(f"  - {i+1}/{len(pairs_to_compare)} | {up1}-{pdb1}:{chain1} vs. {up2}-{pdb2}:{chain2}")
    try:
      mind, meand, aln_mean, rmsd, aln_max, kin_rmsd = get_cre_distance_two_kinases(
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
          mind, meand, aln_mean, rmsd, aln_max, kin_rmsd
        ]
      )
    except Exception as e:
      print(f"There was an error: {str(e)}")
  # Expotar distance summary
  export_distance_summary(distances, folder, job)
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

cli.add_command(calc_kin_cre_distances)
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