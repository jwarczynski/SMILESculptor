from rdkit import Chem
import numpy as np


class SmilesVectorizer(object):
    """SMILES vectorizer and devectorizer, with support for SMILES enumeration (atom order randomization)
    as data augmentation
    
    :parameter charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
    :parameter pad: Length of the vectorization
    :parameter leftpad: Add spaces to the left of the SMILES
    :parameter isomericSmiles: Generate SMILES containing information about stereogenic centers
    :parameter augment: Enumerate the SMILES during transform
    :parameter canonical: use canonical SMILES during transform (overrides enum)
    :parameter binary: Use RDKit binary strings instead of molecule objects
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=10, maxlength=120, 
                 leftpad=True, isomericSmiles=True, augment=True, canonical=False, 
                 startchar = '^', endchar = '$', unknownchar = '?', binary=False):
        #Special Characters
        self.startchar = startchar
        self.endchar = endchar
        self.unknownchar = unknownchar
        
        
        #Vectorization and SMILES options
        self.binary = binary
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.augment = augment
        self.canonical = canonical
        self._pad = pad
        self._maxlength = maxlength
        
        #The characterset
        self._charset = None
        self.charset = charset
        
        #Calculate the dimensions
        self.setdims()

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        #Ensure start and endchars are in the charset
        for char in [self.startchar, self.endchar, self.unknownchar]:
            if char not in charset:
                charset = charset + char
        #Set the hidden properties        
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        self.setdims()
        
    @property
    def maxlength(self):
        return self._maxlength
    
    @maxlength.setter
    def maxlength(self, maxlength):
        self._maxlength = maxlength
        self.setdims()
        
    @property
    def pad(self):
        return self._pad
    
    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self.setdims()
        
    def setdims(self):
        """Calculates and sets the output dimensions of the vectorized molecules from the current settings"""
        self.dims = (self.maxlength + self.pad, self._charlen)
    
        
    def fit(self, mols, extra_chars=[]):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.maxlength and self.charset
        
        :parameter smiles: Numpy array or Pandas series containing smiles as strings
        :parameter extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
        """
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        charset = set("".join(list(smiles))) #Is there a smarter way when the list of SMILES is HUGE!
        self.charset = "".join(charset.union(set(extra_chars)))
        self.maxlength = max([len(smile) for smile in smiles])
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        mol = Chem.MolFromSmiles(smiles)
        nmol = self.randomize_mol(mol)
        return Chem.MolToSmiles(nmol, canonical=self.canonical, isomericSmiles=self.isomericSmiles)
    
    def randomize_mol(self, mol, rng):
        """Performs a randomization of the atom order of an RDKit molecule"""
        ans = list(range(mol.GetNumAtoms()))
        rng.shuffle(ans)
        return Chem.RenumberAtoms(mol,ans)

    def transform(self, mols, augment=None, canonical=None, rng=None):
        """Perform an enumeration (atom order randomization) and vectorization of a Numpy array of RDkit molecules
        
            :parameter mols: The RDKit molecules to transform in a list or array
            :parameter augment: Override the objects .augment setting
            :parameter canonical: Override the objects .canonical setting
            :parameter rng: Numpy random number generator
            :
            
            :output: Numpy array with the vectorized molecules with shape [batch, maxlength+pad, charset]
        """
        #TODO make it possible to use both SMILES, RDKit mols and RDKit binary strings in input
        one_hot =  np.zeros([len(mols)] + list(self.dims), dtype=np.int8)

        #Possibl override object settings
        if augment is None:
            augment = self.augment
        if canonical is None:    
            canonical = self.canonical

        for i,mol in enumerate(mols):
            
            #Fast convert from RDKit binary
            if self.binary: mol = Chem.Mol(mol)
            
            if augment:
                if rng is None:
                    rng = np.random.default_rng()
                mol = self.randomize_mol(mol, rng)
            ss = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=self.isomericSmiles)

            #TODO, Improvement make it robust to too long SMILES strings
            #TODO, Improvement make a "jitter", with random offset within the possible frame
            #TODO, Improvement make it report to many "?"'s

            l = len(ss)
            if self.leftpad:
                offset = self.dims[0]-l-1
            else:
                offset = 1

            for j,c in enumerate(ss):
                charidx = self._char_to_int.get(c, self._char_to_int[self.unknownchar])
                one_hot[i,j+offset,charidx] = 1

            #Pad the start
            one_hot[i,offset-1,self._char_to_int[self.startchar]] = 1
            #Pad the end
            one_hot[i,offset+l:,self._char_to_int[self.endchar]] = 1
            #Pad the space in front of start (Could this lead to funky effects during sampling?)
            #one_hot[i,:offset-1,self._char_to_int[self.endchar]] = 1
            one_hot[i, :offset - 1, self._char_to_int[self.unknownchar]] = 1
                
        return one_hot

      
    def reverse_transform(self, vect, strip=True):
        """ Performs a conversion of a vectorized SMILES to a SMILES strings
        charset must be the same as used for vectorization.

        :parameter vect: Numpy array of vectorized SMILES. Can be a single molecule (2D array) or a batch (3D array).
        :parameter strip: Strip start and end tokens from the SMILES string.
        """
        # Ensure vect is 3D (batch of molecules) for consistent processing
        if vect.ndim == 2:  # Single molecule
            vect = np.expand_dims(vect, axis=0)

        smiles = []
        for v in vect:
            #mask v
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            if strip:
                smile = smile.strip(self.startchar + self.endchar)
            smiles.append(smile)

        # If the input was a single molecule, return a single string instead of a list
        return smiles[0] if len(smiles) == 1 else np.array(smiles)

    @property
    def int_to_char(self):
        return self._int_to_char
