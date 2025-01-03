import numpy as np
import torch
import random
import re

from rdkit import Chem


class BasicSmilesTokenizer(object):
  def __init__(self):
    self.regex_pattern = r"""(\[[^\]]+]|Br?|Cl?|Nb?|In?|Sb?|As|Ai|Ta|Ga|O|P|F|H|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    self.regex = re.compile(self.regex_pattern)

  def tokenize(self, text):
    tokens = [token for token in self.regex.findall(text)]
    return tokens

TOKENIZER = BasicSmilesTokenizer()

class Complete(object):

    def __init__(self, augmentation=False, max_len=38):

        self.augmentation = augmentation
        self.max_len = max_len
        self.char_indices = {
            '#': 0, '(': 1, ')': 2, '-': 3, '/': 4, '1': 5, '2': 6, '3': 7, '4': 8, 
            '5': 9, '6': 10, '=': 11, 'Br': 12, 'C': 13, 'Cl': 14, 'F': 15, 'I': 16,
            'In': 17, 'N': 18, 'O': 19, 'P': 20, 'S': 21, '[17O]': 22, '[AlH-]': 23, 
            '[AlH2-]': 24, '[AlH3-]': 25, '[AsH3-]': 26, '[BH-]': 27, '[BH2-]': 28, 
            '[BH3-]': 29, '[C-]': 30, '[C@@H]': 31, '[C@@]': 32, '[C@H]': 33, 
            '[C@]': 34, '[CH-]': 35, '[CH2-]': 36, '[CH2]': 37, '[CH]': 38, 
            '[C]': 39, '[GaH-]': 40, '[GaH2-]': 41, '[GaH3-]': 42, '[InH-]': 43, 
            '[InH2-]': 44, '[InH3-]': 45, '[N+]': 46, '[N-]': 47, '[N@+]': 48, 
            '[N@@+]': 49, '[N@@]': 50, '[N@]': 51, '[NH+]': 52, '[NH-]': 53, 
            '[NH2+]': 54, '[NH3+]': 55, '[NH]': 56, '[N]': 57, '[NbH3-]': 58, 
            '[O+]': 59, '[O-]': 60, '[O]': 61, '[PH+]': 62, '[PH3-]': 63, '[PH4-]': 64,
            '[S+]': 65, '[SbH3-]': 66, '[Si]': 67, '[TaH3-]': 68, '[c-]': 69, '[cH-]': 70, 
            '[n+]': 71, '[n-]': 72, '[nH+]': 73, '[nH]': 74, '[o+]': 75, '\\': 76, 
            'c': 77, 'n': 78, 'o': 79, '': 80
        }
        
    def __call__(self, data):

        if self.augmentation:
            # SMILES Enumerated examples #1
            mol = Chem.MolFromSmiles(data.smi_original)
            num, atoms_list = range(mol.GetNumAtoms()), mol.GetNumAtoms()
    
            # --------------- SMILES ENUMERATION -------------- #
            random_smi = Chem.MolToSmiles(
                Chem.RenumberAtoms(mol, random.sample(num, atoms_list)),
                canonical=False, isomericSmiles=True
            )
            # ------------------------------------------------- #

            data.random_smi_str1 = random_smi
            tokens = TOKENIZER.tokenize(random_smi)
            random_smi_len = len(tokens)
            random_smi = [self.char_indices[s] for s in tokens]
            random_smi = np.array(self.pad_smile(random_smi, 'right'))
            data.random_smi_len1 = random_smi_len
            data.random_smi1 = torch.from_numpy(random_smi)

            # SMILES Enumerated examples #2
            mol = Chem.MolFromSmiles(data.smi_original)
            num, atoms_list = range(mol.GetNumAtoms()), mol.GetNumAtoms()
    
            # --------------- SMILES ENUMERATION -------------- #
            random_smi = Chem.MolToSmiles(
                Chem.RenumberAtoms(mol, random.sample(num, atoms_list)),
                canonical=False, isomericSmiles=True
            )
            # ------------------------------------------------- #

            data.random_smi_str2 = random_smi
            tokens = TOKENIZER.tokenize(random_smi)
            random_smi_len2 = len(tokens)
            random_smi2 = [self.char_indices[s] for s in tokens]
            random_smi2 = np.array(self.pad_smile(random_smi2, 'right'))
            data.random_smi_len2 = random_smi_len2
            data.random_smi2 = torch.from_numpy(random_smi2)

        tokens = TOKENIZER.tokenize(data.smi_original)
        smi_len = len(tokens)
        smi = [self.char_indices[s] for s in tokens]
        smi = np.array(self.pad_smile(smi, 'right'))
        data.smi_len = smi_len
        data.smi = torch.from_numpy(smi)
        
        return data


    def pad_smile(self, string, padding='right'):
        if len(string) <= self.max_len:
            if padding == 'right':
                return string + [self.char_indices['']] * (self.max_len - len(string))
            elif padding == 'left':
                return [self.char_indices['']] * (self.max_len - len(string)) + string
            elif padding == 'none':
                return string
        else:
          raise ValueError(f'len(SMILES): {len(string)} > max_len: {self.max_len}')

    def get_char_indices(self):
        return self.char_indices
