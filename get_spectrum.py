import numpy as np
import collections
import pickle
import os
import argparse
import openbabel as ob
import pybel
import time
import json

from schnetpack import Properties
from utility_classes import Molecule, ConnectivityCompressor
from utility_functions import run_threaded, print_atom_bond_ring_stats, update_dict
from ase import Atoms
from ase.db import connect

import pybel
from filter_generated import (
    _get_atoms_per_type_str,
    _update_dict,
    check_valency,
    collect_bond_and_ring_stats,
    collect_fingerprints_and_cans,
    filter_unique,
    filter_unique_threaded,
    remove_disconnected,
)


def get_spectrum(pos, atomic_numbers):
    """
    Compute the Raman and IR spectra for a molecule.

    Args:
        pos (numpy.ndarray): positions of the atoms (n_atoms x 3)
        atomic_numbers (numpy.ndarray): types of the atoms (n_atoms)

    Returns:
        numpy.ndarray: Raman spectrum
        numpy.ndarray: IR spectrum
        numpy.ndarray: x-axis values (wavenumber)
    """
    device = torch.device("cpu")
    dtype = torch.float32
    pos_tensor = torch.tensor(pos, device=device, dtype=dtype)
    atomic_numbers_tensor = torch.LongTensor(atomic_numbers).to(device)

    charge_model_ = charge_model(device=device)
    vib_model = nn_vib_analysis(device=device, Linear=False, scale=0.965)

    charge = charge_model_(z=atomic_numbers_tensor, pos=pos_tensor)

    freq, iir, araman = vib_model(z=atomic_numbers_tensor, pos=pos_tensor)

    x_axis = torch.linspace(500, 4000, 3501)
    yir = Lorenz_broadening(freq, iir, c=x_axis, sigma=15).detach().numpy()
    yraman_act = Lorenz_broadening(freq, araman, c=x_axis, sigma=12)
    yraman = get_raman_intensity(x_axis, yraman_act).detach().numpy()

    x = x_axis.detach().numpy()

    return yraman, yir, x


def get_fingerprint(pos, atomic_numbers, use_bits=False, con_mat=None):
    """
    Compute the molecular fingerprint (Open Babel FP2), canonical smiles
    representation, and number of atoms per type (e.g. H2C3O1) of a molecule.

    Args:
        pos (numpy.ndarray): positions of the atoms (n_atoms x 3)
        atomic_numbers (numpy.ndarray): types of the atoms (n_atoms)
        use_bits (bool, optional): set True to return the non-zero bits in the
            fingerprint instead of the pybel.Fingerprint object (default: False)
        con_mat (numpy.ndarray, optional): connectivity matrix of the molecule
            containing the pairwise bond orders between all atoms (n_atoms x n_atoms)
            (can be inferred automatically if not provided, default: None)

    Returns:
        numpy.ndarray: Raman spectrum
        numpy.ndarray: IR spectrum
        numpy.ndarray: x-axis values (wavenumber)
        str: the canonical smiles representation of the molecule
        str: the atom types contained in the molecule followed by number of
            atoms per type, e.g. H2C3O1, ordered by increasing atom type (nuclear
            charge)
    """
    if con_mat is not None:
        mol = Molecule(pos, atomic_numbers, con_mat)
        idc_lists = np.where(con_mat != 0)
        mol._update_bond_orders(idc_lists)
        mol = pybel.Molecule(mol.get_obmol())
    else:
        obmol = ob.OBMol()
        obmol.BeginModify()
        for p, n in zip(pos, atomic_numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        mol = pybel.Molecule(obmol)

    raman_spectrum, ir_spectrum, x_axis = get_spectrum(pos, atomic_numbers)
    return (
        raman_spectrum,
        ir_spectrum,
        x_axis,
        mol.write("can"),
        _get_atoms_per_type_str(atomic_numbers),
    )


# 其他相关的修改可以参考这个模板进行相应替换


def _compare_fingerprints(
    mols,
    train_fps,
    train_idx,
    thresh,
    stats,
    stat_heads,
    print_file=True,
    use_bits=False,
    max_heavy_atoms=9,
):
    """
    Compare fingerprints (or spectra) of generated and training data molecules to update the
    statistics of the generated molecules (to which training/validation/test
    molecule it corresponds, if any).

    Args:
        mols (list of utility_classes.Molecule): generated molecules
        train_fps (dict (str->list of tuple)): dictionary with fingerprints or spectra of
            training/validation/test data as returned by _get_training_fingerprints_dict
        train_idx (list of int): list that maps the index of fingerprints in the
            train_fps dict to indices of the underlying training database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        thresh (tuple of int): tuple containing the number of validation and test
            data molecules (n_validation, n_test)
        stats (numpy.ndarray): statistics of all generated molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: True)
        use_bits (bool, optional): set True if the fingerprint is provided as a list of
            non-zero bits instead of the pybel.Fingerprint object (default: False)
        max_heavy_atoms (int, optional): the maximum number of heavy atoms in the
            training data set (i.e. 9 for qm9, default: 9)

    Returns:
        dict (str->numpy.ndarray): dictionary containing the updated statistics under
            the key 'stats'
    """
    idx_known = stat_heads.index("known")
    idx_equals = stat_heads.index("equals")
    idx_val = stat_heads.index("valid")
    n_val_mols, n_test_mols = thresh
    # get indices of valid molecules
    idcs = np.where(stats[:, idx_val] == 1)[0]
    if not print_file:
        print(f"0.00%", end="", flush=True)
    for i, idx in enumerate(idcs):
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        # for now the molecule is considered to be new
        stats[idx, idx_known] = 0
        if np.sum(mol.numbers != 1) > max_heavy_atoms:
            continue  # cannot be in dataset
        if mol_key in train_fps:
            for fp_train in train_fps[mol_key]:
                # compare spectrum
                if np.allclose(
                    mol.raman_spectrum, fp_train[0], atol=0.1
                ) and np.allclose(mol.ir_spectrum, fp_train[1], atol=0.1):
                    # compare canonical smiles representation
                    if (
                        mol.get_can() == fp_train[2]
                        or mol.get_mirror_can() == fp_train[2]
                    ):
                        # store index of match
                        j = fp_train[-1]
                        stats[idx, idx_equals] = train_idx[j]
                        if j >= len(train_idx) - np.sum(thresh):
                            if j > len(train_idx) - n_test_mols:
                                stats[idx, idx_known] = 3  # equals test data
                            else:
                                stats[idx, idx_known] = 2  # equals validation data
                        else:
                            stats[idx, idx_known] = 1  # equals training data
                        break
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(f"{100 * (i + 1) / len(idcs):.2f}%", end="\r", flush=True)
    if not print_file:
        print("\033[K", end="", flush=True)
    return {"stats": stats}


def filter_new(
    mols, stats, stat_heads, model_path, train_data_path, print_file=False, n_threads=0
):
    """
    Check whether generated molecules correspond to structures in the training database
    used for either training, validation, or as test data and update statistics array of
    generated molecules accordingly.

    Args:
        mols (list of utility_classes.Molecule): generated molecules
        stats (numpy.ndarray): statistics of all generated molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        model_path (str): path to the folder containing the trained model used to
            generate the molecules
        train_data_path (str): full path to the training database
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: False)
        n_threads (int, optional): number of additional threads to use (default: 0)

    Returns:
        numpy.ndarray: updated statistics of all generated molecules (stats['known']
        is 0 if a generated molecule does not correspond to a structure in the
        training database, it is 1 if it corresponds to a training structure,
        2 if it corresponds to a validation structure, and 3 if it corresponds to a
        test structure, stats['equals'] is -1 if stats['known'] is 0 and otherwise
        holds the index of the corresponding training/validation/test structure in
        the database at train_data_path)
    """
    print(f"\n\n2. Checking which molecules are new...")
    idx_known = stat_heads.index("known")

    # load training data
    dbpath = train_data_path
    if not os.path.isfile(dbpath):
        print(
            f"The provided training data base {dbpath} is no file, please specify "
            f"the correct path (including the filename and extension)!"
        )
        raise FileNotFoundError
    print(f"Using data base at {dbpath}...")

    split_file = os.path.join(model_path, "split.npz")
    if not os.path.exists(split_file):
        raise FileNotFoundError
    S = np.load(split_file)
    train_idx = S["train_idx"]
    val_idx = S["val_idx"]
    test_idx = S["test_idx"]
    train_idx = np.append(train_idx, val_idx)
    train_idx = np.append(train_idx, test_idx)

    # check if subset was used (and restrict indices accordingly)
    train_args_path = os.path.join(model_path, f"args.json")
    with open(train_args_path) as handle:
        train_args = json.loads(handle.read())
    if "subset_path" in train_args:
        if train_args["subset_path"] is not None:
            subset = np.load(train_args["subset_path"])
            train_idx = subset[train_idx]

    print("\nComputing spectra of training data...")
    start_time = time.time()
    if n_threads <= 0:
        train_spectra = _get_training_spectra(
            dbpath, train_idx, print_file, use_con_mat=True
        )
    else:
        train_spectra = {"spectra": [None for _ in range(len(train_idx))]}
        run_threaded(
            _get_training_spectra,
            {"train_idx": train_idx},
            {"dbpath": dbpath, "use_con_mat": True},
            train_spectra,
            exclusive_kwargs={"print_file": print_file},
            n_threads=n_threads,
        )
    train_spectra_dict = _get_training_spectra_dict(train_spectra["spectra"])
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(
        f'...{len(train_spectra["spectra"])} spectra computed '
        f"in {h:d}h{m:02d}m{s:02d}s!"
    )

    print("\nComparing spectra...")
    start_time = time.time()
    if n_threads <= 0:
        results = _compare_spectra(
            mols,
            train_spectra_dict,
            train_idx,
            [len(val_idx), len(test_idx)],
            stats.T,
            stat_heads,
            print_file,
        )
    else:
        results = {"stats": stats.T}
        run_threaded(
            _compare_spectra,
            {"mols": mols, "stats": stats.T},
            {
                "train_idx": train_idx,
                "train_spectra": train_spectra_dict,
                "thresh": [len(val_idx), len(test_idx)],
                "stat_heads": stat_heads,
            },
            results,
            exclusive_kwargs={"print_file": print_file},
            n_threads=n_threads,
        )
    stats = results["stats"].T
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f"... needed {h:d}h{m:02d}m{s:02d}s.")
    print(
        f"Number of new molecules: "
        f"{sum(stats[idx_known] == 0)+sum(stats[idx_known] == 3)}"
    )
    print(
        f"Number of molecules matching training data: " f"{sum(stats[idx_known] == 1)}"
    )
    print(
        f"Number of molecules matching validation data: "
        f"{sum(stats[idx_known] == 2)}"
    )
    print(f"Number of molecules matching test data: " f"{sum(stats[idx_known] == 3)}")

    return stats


def _get_training_spectra(dbpath, train_idx, print_file=True, use_con_mat=False):
    """
    Get the spectra (Raman and IR) of all molecules in the training database.

    Args:
        dbpath (str): path to the training database
        train_idx (list of int): list containing the indices of training, validation,
            and test molecules in the database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        use_con_mat (bool, optional): set True to use pre-computed connectivity
            matrices (need to be stored in the training database in compressed format
            under the key 'con_mat', default: False)

    Returns:
        dict (str->list of tuple): dictionary with list of tuples under the key
        'spectra' containing the Raman spectrum, IR spectrum, and the atoms per type string
        of each molecule listed in train_idx (preserving the order)
    """
    train_spectra = []
    if use_con_mat:
        compressor = ConnectivityCompressor()
    with connect(dbpath) as conn:
        if not print_file:
            print("0.00%", end="\r", flush=True)
        for i, idx in enumerate(train_idx):
            idx = int(idx)
            row = conn.get(idx + 1)
            at = row.toatoms()
            pos = at.positions
            atomic_numbers = at.numbers
            if use_con_mat:
                con_mat = compressor.decompress(row.data["con_mat"])
            else:
                con_mat = None
            train_spectra += [get_spectrum(pos, atomic_numbers)]
            if (i % 100 == 0 or i + 1 == len(train_idx)) and not print_file:
                print("\033[K", end="\r", flush=True)
                print(f"{100 * (i + 1) / len(train_idx):.2f}%", end="\r", flush=True)
    return {"spectra": train_spectra}


def _get_training_spectra_dict(spectra):
    """
    Convert a list of spectra into a dictionary where a string describing the
    number of types in each molecules (e.g. H2C3O1, ordered by increasing nuclear
    charge) is used as a key (allows for faster comparison of molecules as only those
    made of the same atoms can be identical).

    Args:
        spectra (list of tuple): list containing tuples as returned by the get_spectrum
            function (holding the Raman spectrum, IR spectrum, and the atoms per type string)

    Returns:
        dict (str->list of tuple): dictionary containing lists of tuples holding the
            molecular Raman spectrum, IR spectrum, and the index of the molecule in the input list
            using the atoms per type string of the molecules as key (such that spectrum tuples of
            all molecules with the exact same atom composition, e.g. H2C3O1, are stored together in one list)
    """
    spectra_dict = {}
    for i, spec in enumerate(spectra):
        spectra_dict = _update_dict(spectra_dict, key=spec[-1], val=spec[:-1] + (i,))
    return spectra_dict


def _compare_spectra(
    mols,
    train_spectra,
    train_idx,
    thresh,
    stats,
    stat_heads,
    print_file=True,
    max_heavy_atoms=9,
):
    """
    Compare spectra of generated and training data molecules to update the
    statistics of the generated molecules (to which training/validation/test
    molecule it corresponds, if any).

    Args:
        mols (list of utility_classes.Molecule): generated molecules
        train_spectra (dict (str->list of tuple)): dictionary with spectra of
            training/validation/test data as returned by _get_training_spectra_dict
        train_idx (list of int): list that maps the index of spectra in the
            train_spectra dict to indices of the underlying training database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        thresh (tuple of int): tuple containing the number of validation and test
            data molecules (n_validation, n_test)
        stats (numpy.ndarray): statistics of all generated molecules where columns
            correspond to molecules and rows correspond to available statistics
            (n_statistics x n_molecules)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: True)
        max_heavy_atoms (int, optional): the maximum number of heavy atoms in the
            training data set (i.e. 9 for qm9, default: 9)

    Returns:
        dict (str->numpy.ndarray): dictionary containing the updated statistics under
            the key 'stats'
    """
    idx_known = stat_heads.index("known")
    idx_equals = stat_heads.index("equals")
    idx_val = stat_heads.index("valid")
    n_val_mols, n_test_mols = thresh
    # get indices of valid molecules
    idcs = np.where(stats[:, idx_val] == 1)[0]
    if not print_file:
        print(f"0.00%", end="", flush=True)
    for i, idx in enumerate(idcs):
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        # for now the molecule is considered to be new
        stats[idx, idx_known] = 0
        if np.sum(mol.numbers != 1) > max_heavy_atoms:
            continue  # cannot be in dataset
        if mol_key in train_spectra:
            for spec_train in train_spectra[mol_key]:
                # compare spectrum
                if np.allclose(
                    mol.raman_spectrum, spec_train[0], atol=0.1
                ) and np.allclose(mol.ir_spectrum, spec_train[1], atol=0.1):
                    # compare canonical smiles representation
                    if (
                        mol.get_can() == spec_train[2]
                        or mol.get_mirror_can() == spec_train[2]
                    ):
                        # store index of match
                        j = spec_train[-1]
                        stats[idx, idx_equals] = train_idx[j]
                        if j >= len(train_idx) - np.sum(thresh):
                            if j > len(train_idx) - n_test_mols:
                                stats[idx, idx_known] = 3  # equals test data
                            else:
                                stats[idx, idx_known] = 2  # equals validation data
                        else:
                            stats[idx, idx_known] = 1  # equals training data
                        break
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(f"{100 * (i + 1) / len(idcs):.2f}%", end="\r", flush=True)
    if not print_file:
        print("\033[K", end="", flush=True)
    return {"stats": stats}


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "data_path",
        help="Path to generated molecules in .mol_dict format, "
        'a database called "generated_molecules.db" with the '
        "filtered molecules along with computed statistics "
        '("generated_molecules_statistics.npz") will be '
        "stored in the same directory as the input file/s "
        "(if the path points to a directory, all .mol_dict "
        "files in the directory will be merged and filtered "
        "in one pass)",
    )
    main_parser.add_argument(
        "--train_data_path",
        help="Path to training data base (if provided, "
        "generated molecules can be compared/matched with "
        "those in the training data set)",
        default=None,
    )
    main_parser.add_argument(
        "--model_path",
        help="Path of directory containing the model that "
        "generated the molecules. It should contain a "
        "split.npz file with training data splits and a "
        "args.json file with the arguments used during "
        "training (if this and --train_data_path "
        "are provided, the generated molecules will be "
        "filtered for new structures which were not included "
        "in the training or validation data)",
        default=None,
    )
    main_parser.add_argument(
        "--valence",
        default=[1, 1, 6, 4, 7, 3, 8, 2, 9, 1],
        type=int,
        nargs="+",
        help="the valence of atom types in the form "
        "[type1 valence type2 valence ...] "
        "(default: %(default)s)",
    )
    main_parser.add_argument(
        "--filters",
        type=str,
        nargs="*",
        default=["valence", "disconnected", "unique"],
        choices=["valence", "disconnected", "unique"],
        help="Select the filters applied to identify "
        "invalid molecules (default: %(default)s)",
    )
    main_parser.add_argument(
        "--store",
        type=str,
        nargs="+",
        default=["valid", "connectivity"],
        choices=["all", "valid", "new", "connectivity", "fingerprint", "process"],
        help="How much information shall be stored "
        'after filtering: \n"all" keeps all '
        "generated molecules and statistics,"
        '\n"valid" keeps only valid molecules and their '
        "statistics and,\n"
        '"new" furthermore discards all validly '
        "generated molecules that match training "
        'data (corresponds to "valid" if '
        "model_path is not provided), "
        '\nadditionally providing "connectivity" and/or '
        '"fingerprint" will store the '
        "corresponding connectivity matrices and "
        "fingerprint information (i.e. the bits set in the "
        "fingerprint and the canonical smiles strings of "
        "the usual and mirrored molecules) in the data "
        'base. \nWhen "process" is provided, '
        "the necessary information to reproduce the "
        "generation process is stored in the data base ("
        "i.e. the predicted distributions and the focus at "
        "each step). Note that this is only possible, "
        "if this information was tracked and stored during "
        "generation. (default: %(default)s)",
    )
    main_parser.add_argument(
        "--print_file",
        help="Use to limit the printing if results are "
        "written to a file instead of the console ("
        "e.g. if running on a cluster)",
        action="store_true",
    )
    main_parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads used (set to 0 to run "
        "everything sequentially in the main thread,"
        " default: %(default)s)",
    )

    return main_parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file

    # read input file or fuse dictionaries if data_path is a folder
    if not os.path.isdir(args.data_path):
        if not os.path.isfile(args.data_path):
            print(
                f"\n\nThe specified data path ({args.data_path}) is neither a file "
                f"nor a directory! Please specify a different data path."
            )
            raise FileNotFoundError
        else:
            with open(args.data_path, "rb") as f:
                res = pickle.load(f)  # read input file
            target_db = os.path.join(
                os.path.dirname(args.data_path), "generated_molecules.db"
            )
    else:
        print(f"\n\nFusing .mol_dict files in folder {args.data_path}...")
        mol_files = [f for f in os.listdir(args.data_path) if f.endswith(".mol_dict")]
        if len(mol_files) == 0:
            print(
                f"Could not find any .mol_dict files at {args.data_path}! Please "
                f"specify a different data path!"
            )
            raise FileNotFoundError
        res = {}
        for file in mol_files:
            with open(os.path.join(args.data_path, file), "rb") as f:
                cur_res = pickle.load(f)
                update_dict(res, cur_res)
        res = dict(sorted(res.items()))  # sort dictionary keys
        print(f"...done!")
        target_db = os.path.join(args.data_path, "generated_molecules.db")

    # compute array with valence of provided atom types
    max_type = max(args.valence[::2])
    valence = np.zeros(max_type + 1, dtype=int)
    valence[args.valence[::2]] = args.valence[1::2]

    # print the chosen settings
    valence_str = ""
    for i in range(max_type + 1):
        if valence[i] > 0:
            valence_str += f"type {i}: {valence[i]}, "
    filters = []
    if "valence" in args.filters:
        filters += ["valency"]
    if "disconnected" in args.filters:
        filters += ["connectedness"]
    if "unique" in args.filters:
        filters += ["uniqueness"]
    if len(filters) >= 3:
        edit = ", "
    else:
        edit = " "
    for i in range(len(filters) - 1):
        filters[i] = filters[i] + edit
    if len(filters) >= 2:
        filters = filters[:-1] + ["and "] + filters[-1:]
    string = "".join(filters)
    print(f"\n\n1. Filtering molecules according to {string}...")
    print(f"\nTarget valence:\n{valence_str[:-2]}\n")

    # initial setup of array for statistics and some counters
    n_generated = 0
    n_valid = 0
    n_non_unique = 0
    stat_heads = [
        "n_atoms",
        "id",
        "original_id",
        "valid",
        "duplicating",
        "n_duplicates",
        "known",
        "equals",
        "C",
        "N",
        "O",
        "F",
        "H",
        "H1C",
        "H1N",
        "H1O",
        "C1C",
        "C2C",
        "C3C",
        "C1N",
        "C2N",
        "C3N",
        "C1O",
        "C2O",
        "C1F",
        "N1N",
        "N2N",
        "N1O",
        "N2O",
        "N1F",
        "O1O",
        "O1F",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R>8",
    ]
    stats = np.empty((len(stat_heads), 0))
    all_mols = []
    connectivity_compressor = ConnectivityCompressor()

    # construct connectivity matrix and fingerprints for filtering
    start_time = time.time()
    for n_atoms in res:
        if not isinstance(n_atoms, int) or n_atoms == 0:
            continue

        prog_str = lambda x: f"Checking {x} for molecules of length {n_atoms}"
        work_str = "valence" if "valence" in args.filters else "dictionary"
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(prog_str(work_str) + " (0.00%)", end="\r", flush=True)
        else:
            print(prog_str(work_str), flush=True)

        d = res[n_atoms]
        all_pos = d[Properties.R]
        all_numbers = d[Properties.Z]
        n_mols = len(all_pos)

        # check valency
        if args.threads <= 0:
            results = check_valency(
                all_pos,
                all_numbers,
                valence,
                "valence" in args.filters,
                print_file,
                prog_str(work_str),
            )
        else:
            results = {
                "connectivity": np.zeros((n_mols, n_atoms, n_atoms)),
                "mols": [None for _ in range(n_mols)],
                "valid": np.ones(n_mols, dtype=bool),
            }
            results = run_threaded(
                check_valency,
                {"positions": all_pos, "numbers": all_numbers},
                {
                    "valence": valence,
                    "filter_by_valency": "valence" in args.filters,
                    "picklable_mols": True,
                    "prog_str": prog_str(work_str),
                },
                results,
                n_threads=args.threads,
                exclusive_kwargs={"print_file": print_file},
            )
        connectivity = results["connectivity"]
        mols = results["mols"]
        valid = results["valid"]

        # detect molecules with disconnected parts if desired
        if "disconnected" in args.filters:
            if not print_file:
                print("\033[K", end="\r", flush=True)
                print(prog_str("connectedness") + "...", end="\r", flush=True)
            if args.threads <= 0:
                valid = remove_disconnected(connectivity, valid)["valid"]
            else:
                results = {"valid": valid}
                run_threaded(
                    remove_disconnected,
                    {"connectivity_batch": connectivity, "valid": valid},
                    {},
                    results,
                    n_threads=args.threads,
                )
                valid = results["valid"]

        # identify molecules with identical fingerprints
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(prog_str("uniqueness") + "...", end="\r", flush=True)
        if args.threads <= 0:
            still_valid, duplicating, duplicate_count = filter_unique(
                mols, valid, use_bits=False
            )
        else:
            still_valid, duplicating, duplicate_count = filter_unique_threaded(
                mols,
                valid,
                n_threads=args.threads,
                n_mols_per_thread=5,
                print_file=print_file,
                prog_str=prog_str("uniqueness"),
            )
        n_non_unique += np.sum(duplicate_count)
        if "unique" in args.filters:
            valid = still_valid  # remove non-unique from valid if desired

        # store connectivity matrices
        d.update(
            {
                "connectivity": connectivity_compressor.compress_batch(connectivity),
                "valid": valid,
            }
        )

        # if desired, store fingerprint information
        if "fingerprint" in args.store:
            if args.threads <= 0:
                results = collect_fingerprints_and_cans(mols)
            else:
                results = {
                    "fingerprint_bits": [],
                    "canonical_smiles": [],
                    "mirrored_canonical_smiles": [],
                }
                run_threaded(
                    collect_fingerprints_and_cans,
                    {"mols": mols},
                    {},
                    results,
                    n_threads=args.threads,
                )
            d.update(results)

        # collect statistics of generated data
        n_generated += len(valid)
        n_valid += np.sum(valid)
        n_of_types = [np.sum(all_numbers == i, axis=1) for i in [6, 7, 8, 9, 1]]
        stats_new = np.stack(
            (
                np.ones(len(valid)) * n_atoms,  # n_atoms
                np.arange(0, len(valid)),  # id
                np.arange(0, len(valid)),  # original id
                valid,  # valid
                duplicating,  # id of duplicated molecule
                duplicate_count,  # number of duplicates
                -np.ones(len(valid)),  # known
                -np.ones(len(valid)),  # equals
                *n_of_types,  # n_atoms per type
                *np.zeros((19, len(valid))),  # n_bonds per type pairs
                *np.zeros((7, len(valid))),  # ring counts for 3-8 & >8
            ),
            axis=0,
        )
        stats = np.hstack((stats, stats_new))
        all_mols += mols

    if not print_file:
        print("\033[K", end="\r", flush=True)
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f"Needed {h:d}h{m:02d}m{s:02d}s.")

    if args.threads <= 0:
        results = collect_bond_and_ring_stats(all_mols, stats.T, stat_heads)
    else:
        results = {"stats": stats.T}
        run_threaded(
            collect_bond_and_ring_stats,
            {"mols": all_mols, "stats": stats.T},
            {"stat_heads": stat_heads},
            results=results,
            n_threads=args.threads,
        )
    stats = results["stats"].T

    # store statistics
    res.update(
        {
            "n_generated": n_generated,
            "n_valid": n_valid,
            "stats": stats,
            "stat_heads": stat_heads,
        }
    )

    print(
        f"Number of generated molecules: {n_generated}\n"
        f"Number of duplicate molecules: {n_non_unique}"
    )
    if "unique" in args.filters:
        print(f"Number of unique and valid molecules: {n_valid}")
    else:
        print(f"Number of valid molecules (including duplicates): {n_valid}")

    # filter molecules which were seen during training
    if args.model_path is not None:
        stats = filter_new(
            all_mols,
            stats,
            stat_heads,
            args.model_path,
            args.train_data_path,
            print_file=print_file,
            n_threads=args.threads,
        )
        res.update({"stats": stats})

    # shrink results dictionary (remove invalid attempts, known molecules and
    # connectivity matrices if desired)
    if "all" not in args.store:
        shrunk_res = {}
        shrunk_stats = np.empty((len(stats), 0))
        i = 0
        for key in res:
            if isinstance(key, str):
                shrunk_res[key] = res[key]
                continue
            if key == 0:
                continue
            d = res[key]
            start = i
            end = i + len(d["valid"])
            idcs = np.where(d["valid"])[0]
            if len(idcs) < 1:
                i = end
                continue
            # shrink stats
            idx_id = stat_heads.index("id")
            idx_known = stat_heads.index("known")
            new_stats = stats[:, start:end]
            if "new" in args.store and args.model_path is not None:
                idcs = idcs[np.where(new_stats[idx_known, idcs] == 0)[0]]
            new_stats = new_stats[:, idcs]
            new_stats[idx_id] = np.arange(len(new_stats[idx_id]))  # adjust ids
            shrunk_stats = np.hstack((shrunk_stats, new_stats))
            # shrink positions and atomic numbers
            shrunk_res[key] = {
                Properties.R: d[Properties.R][idcs],
                Properties.Z: d[Properties.Z][idcs],
            }
            # store additional information if desired
            additional_information = []
            if "connectivity" in args.store:
                additional_information += ["connectivity"]
            if "fingerprint" in args.store:
                additional_information += [
                    "fingerprint_bits",
                    "canonical_smiles",
                    "mirrored_canonical_smiles",
                ]
            if "process" in args.store:
                additional_information += [
                    "focus",
                    "type_chosen",
                    "type_prediction",
                    "dist_predictions",
                ]
            shrunk_res[key].update(
                {info: [d[info][k] for k in idcs] for info in additional_information}
            )
            i = end

        shrunk_res["stats"] = shrunk_stats
        res = shrunk_res

    # store results in new database
    # get filename that is not yet taken for db
    if os.path.isfile(target_db):
        file_name, _ = os.path.splitext(target_db)
        expand = 0
        while True:
                expand += 1
                new_file_name = file_name + "_" + str(expand)
                if os.path.isfile(new_file_name + ".db"):
                    continue
                else:
                    target_db = new_file_name + ".db"
                    break
        # open db
        with connect(target_db) as conn:
            # store metadata
            conn.metadata = {
                "n_generated": int(n_generated),
                "n_non_unique": int(n_non_unique),
                "n_valid": int(n_valid),
                "non_unique_removed_from_valid": "unique" in args.filters,
            }
            # store molecules
            for n_atoms in res:
                if isinstance(n_atoms, str) or n_atoms == 0:
                    continue
                d = res[n_atoms]
                for i in range(len(d[Properties.R])):
                    at = Atoms(d[Properties.Z][i], positions=d[Properties.R][i])
                    data = {}
                    if "connectivity" in args.store:
                        data.update({"con_mat": d["connectivity"][i]})
                    if "fingerprint" in args.store:
                        data.update(
                            {
                                "fingerprint_bits": np.array(
                                    list(d["fingerprint_bits"][i])
                                ),
                                "canonical_smiles": str(d["canonical_smiles"][i]),
                                "mirrored_canonical_smiles": str(
                                    d["mirrored_canonical_smiles"][i]
                                ),
                            }
                        )
                    if "process" in args.store:
                        data.update(
                            {
                                info: d[info][i]
                                for info in [
                                    "focus",
                                    "type_chosen",
                                    "type_prediction",
                                    "dist_predictions",
                                ]
                            }
                        )
                    conn.write(at, data=data)
    
        # store gathered statistics in separate file
        np.savez_compressed(
            os.path.splitext(target_db)[0] + f"_statistics.npz",
            stats=res["stats"],
            stat_heads=res["stat_heads"],
        )
    
        # print average atom, bond, and ring count statistics of generated molecules
        # stored in the database and of the training molecules
        print_atom_bond_ring_stats(target_db, args.model_path, args.train_data_path)
