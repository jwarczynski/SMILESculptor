import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


from .SmilesVectorizer import SmilesVectorizer


def one_hot_encode(index, length):
    """Create a one-hot encoded vector.

    Args:
        index (int): The index to set to 1.
        length (int): The total length of the vector.

    Returns:
        np.ndarray: One-hot encoded vector.
    """
    vector = np.zeros(length, dtype=int)  # Create a zero vector
    vector[index] = 1  # Set the specified index to 1
    return vector


def convert_to_ohe(encoded, num_classes):
    """Convert an encoded array to one-hot encoding.

    Args:
        encoded (np.ndarray): Encoded array.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded array.
    """
    encoded_shape = (encoded.shape[0], encoded.shape[1], num_classes)
    print(f'One hot encoded shape: {encoded_shape}')

    one_hot_encoded = np.zeros(encoded_shape, dtype=int)

    for i in range(encoded.shape[0]):
        for j in range(encoded.shape[1]):
            index = encoded[i, j]
            one_hot_encoded[i, j] = one_hot_encode(index, num_classes)

    return one_hot_encoded


def convert_to_ohe_and_save(laod_path, save_path, int_to_char_path):
    encoded = np.load(laod_path)
    print(f"Original moles shape: {encoded.shape}")

    with open(int_to_char_path, 'rb') as file:
        int_to_char = pickle.load(file)

    num_classes = len(int_to_char)
    print(f"Number of classes (one-hot length): {num_classes}")

    ohe = convert_to_ohe(encoded, num_classes)
    np.save(save_path, ohe)
    print(f"Saved one-hot encoded array with shape: {ohe.shape} to {save_path}")


def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    mol = Chem.MolFromSmiles(smiles)

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)

    # if many fragments, get the "parent" (the actual mol we are interested in)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.

    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)


def remove_stereochemistry(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        Chem.RemoveStereochemistry(molecule)
        return Chem.MolToSmiles(molecule)
    else:
        raise ValueError("Invalid SMILES string")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default="../../data")
    parser.add_argument("-sp", "--moles-save-path", type=str, required=True)
    parser.add_argument("-itc", "--int-to-char-path", type=str, required=True)
    parser.add_argument("-vp", "--vectorizer-path", type=str, required=True)
    parser.add_argument("-rs", "--random-state", type=int, default=42)
    parser.add_argument("-ns", "--num-samples", type=int, default=1_000_000)
    parser.add_argument("-ohe", "--one-hot-encode", action='store_true')
    parser.add_argument("-a", "--augment", action='store_true')
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    MOLES_SAVE_PATH = Path(args.moles_save_path)
    IDX_TO_CHR_PATH = Path(args.int_to_char_path)
    NUM_SAMPLES = args.num_samples
    RANDOM_STATE = args.random_state
    ONE_HOT = args.one_hot_encode
    VECT_PATH = args.vectorizer_path
    AUGMENT = args.augment

    print(args)

    df = pd.read_csv(DATA_DIR / 'zinc22_random.zip', compression='zip')

    df['SMILES_st'] = df['SMILES'].map(remove_stereochemistry)
    df = df.drop_duplicates(subset='SMILES_st')
    
    num_samples = min(NUM_SAMPLES, df.shape[0])
    df_sample = df.sample(num_samples, random_state=RANDOM_STATE)

    mols = [Chem.MolFromSmiles(item) for item in df_sample['SMILES_st']]

    sv = SmilesVectorizer(augment=AUGMENT)
    sv.fit(mols)

    print(sv.charset)
    print(f'chaserset length: {len(sv.charset)}')

    enc_mols = sv.transform(mols)
    print(f'encoded moles shape: {enc_mols.shape}')

    np.save(MOLES_SAVE_PATH, enc_mols)
    print(f'Saved encoded moles to {MOLES_SAVE_PATH}')

    with open(IDX_TO_CHR_PATH, 'wb') as f:
        pickle.dump(sv.int_to_char, f)
        print(f'Saved index to character mapping to {IDX_TO_CHR_PATH}')

    with open(VECT_PATH, 'wb') as f:
        pickle.dump(sv, f)
        print(f'Saved vectorizer to {VECT_PATH}')
