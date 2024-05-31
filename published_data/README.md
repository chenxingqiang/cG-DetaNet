# Generated molecules and pretrained models

Here we provide links to the molecules generated with cG-SchNet in the studies reported in the paper. This allows for reproduction of the shown graphs and statistics, as well as further analysis of the obtained molecules. Furthermore, we provide two pretrained cG-SchNet models that were used in our experiments, which enables sampling of additional molecules.

## Data bases with generated molecules
A zip-file containing the generated molecules can be found under [DOI 10.14279/depositonce-14977](http://dx.doi.org/10.14279/depositonce-14977).
It includes five folders with molecules generated by five cG-SchNet models conditioned on different combinations of properties.
They appear in the same order as in the paper, i.e. the first model is conditioned on isotropic polarizability, the second is conditioned on fingerprints, the third is conditioned on HOMO-LUMO gap and atomic composition, the fourth is conditioned on relative atomic energy and atomic composition, and the fifth is conditioned on HOMO-LUMO gap and relative atomic energy. Each folder contains several data bases following the naming convention _\<target values\>\_\<type\>.db_, where _\<target values\>_ lists the conditioning target values (separated with underscores) and _\<type\>_ is either _generated_ for raw, generated structures or _relaxed_ for the relaxed counterparts (wherever we computed them). For example _/4\_comp\_relenergy/c7o2h10\_-0.1\_relaxed.db_ contains relaxed molecules that were generated by the fourth cG-SchNet model (conditioned on atomic composition and relative atomic energy) using the composition c7o2h10 and a relative atomic energy of -0.1 eV as targets during sampling. Relaxation was carried out at the same level of theory used for the QM9 training data. Please refer to the publication for details on the procedure.

For help with the content of each data base, please call the script _data\_base\_help.py_ which will print the metadata of a selected data base and help to get started:

```
python ./cG-SchNet/published_data/data_base_help.py <path-to-db>
```

For example, the output for the data base _/4\_comp\_relenergy/c7o2h10\_-0.1\_relaxed.db_ is as following:

```
INFO for data base at <path-to-db>
===========================================================================================================
Contains 2719 C7O2H10 isomers generated by cG-SchNet and subsequently relaxed with DFT calculations.
All reported energy values are in eV and a systematic offset between the ORCA reference calculations and the QM9 reference values was approximated and compensated. For reference, the uncompensated energy values directly obtained with ORCA are also provided.
The corresponding raw, generated version of each structure can be found in the data base of generated molecules using the provided "gen_idx" (caution, it starts from 0 whereas ase loads data bases counting from 1).
The field "rmsd" holds the root mean square deviation between the atom positions of the raw, generated structure and the relaxed equilibrium molecule in Å (it includes hydrogen atoms).
The field "changed" is 0 if the connectivity of the molecule before and after relaxation is identical (i.e. no bonds were broken or newly formed) and 1 if it did change.
The field "known_relaxed" is 0, 3, or 6 to mark novel isomers, novel stereo-isomers, and unseen isomers (i.e. isomers resembling test data), respectively.
Originally, all 3349 unique and valid C7O2H10 isomers among 100k molecules generated by cG-SchNet were chosen for relaxation. 7 of these did not converge to a valid configuration and 630 converged to equilibrium configurations already covered by other generated isomers (i.e. they were duplicate structures) and were therefore removed.
===========================================================================================================

For example, here is the data stored with the first three molecules:
0: {'gen_idx': 0, 'computed_relative_atomic_energy': -0.11081359089212128, 'computed_energy_U0': -11512.699587841627, 'computed_energy_U0_uncompensated': -11512.578122651728, 'rmsd': 0.2492194704871389, 'changed': 0, 'known_relaxed': 3}
1: {'gen_idx': 1, 'computed_relative_atomic_energy': -0.13234655336327705, 'computed_energy_U0': -11513.108714128579, 'computed_energy_U0_uncompensated': -11512.98724893868, 'rmsd': 0.48381233235411153, 'changed': 0, 'known_relaxed': 3}
2: {'gen_idx': 2, 'computed_relative_atomic_energy': -0.11825264882338615, 'computed_energy_U0': -11512.84092994232, 'computed_energy_U0_uncompensated': -11512.719464752421, 'rmsd': 0.21804191287675742, 'changed': 0, 'known_relaxed': 3}

You can load and access the molecules and accompanying data by connecting to the data base with ASE, e.g. using the following python code snippet:
from ase.db import connect
with connect(<path-to-db>) as con:
 	row = con.get(1)  # load the first molecule, 1-based indexing
 	R = row.positions  # positions of atoms as 3d coordinates
 	Z = row.numbers  # list of atomic numbers
 	data = row.data  # dictionary of data stored with the molecule

You can visualize the molecules in the data base with ASE from the command line by calling:
ase gui <path-to-db>
```

Note that references to molecules from the QM9 data set always correspond to the indices _after_ removing the invalid molecules listed in [the file of invalid structures](https://github.com/atomistic-machine-learning/cG-SchNet/blob/main/splits/qm9_invalid.txt). These structures were automatically removed if QM9 was downloaded with the data script in this repository (e.g. by starting [model training](https://github.com/atomistic-machine-learning/cG-SchNet#training-a-model)). The resulting data base with corresponding indices can be found in your data directory: ```<data-directory-path>/qm9gen.db```.

## Pretrained models
A zip-file containing two pretrained cG-SchNet models can be found under [DOI 10.14279/depositonce-14978](http://dx.doi.org/10.14279/depositonce-14978). The archive consists of two folders, where _comp\_relenergy_ hosts the model that was conditioned on atomic composition and relative atomic energy and used for the study described in Fig. 4 in the paper. The other model was conditioned on the HOMO-LUMO gap and relative atomic energy and used in the study described in Fig. 5 in the paper.

In order to generate molecules with the pretrained models, simply extract the folders into your model directory and adapt the call for generating molecules described in the [main readme](https://github.com/atomistic-machine-learning/cG-SchNet#generating-molecules) accordingly. For example, you can generate 20k molecules with the composition c7o2h10 and a relative atomic energy of -0.1 eV as targets with:

```
python ./cG-SchNet/gschnet_cond_script.py generate gschnet ./models/comp_relenergy/ 20000 --conditioning "composition 10 7 0 2 0; n_atoms 19; relative_atomic_energy -0.1" --cuda
```

Note that the model takes a string with three conditions as input to the --conditioning argument: the number of atoms of each type in the order h c n o f, the total number of atoms, and the relative atomic energy value, each separated with a semicolon.
Similarly, you can generate 20k molecules with a HOMO-LUMO gap of 4 eV and relative atomic energy of -0.2 eV as targets with the other model:

```
python ./cG-SchNet/gschnet_cond_script.py generate gschnet ./models/gap_relenergy/ 20000 --conditioning "gap 4.0; relative_atomic_energy -0.2" --cuda
```

The second model takes two conditions, the gap value and the energy value, as targets.
For more details on the generation of molecules and subsequent filtering, please refer to the [main readme](https://github.com/atomistic-machine-learning/cG-SchNet#filtering-and-analysis-of-generated-molecules).