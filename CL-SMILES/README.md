# Enhancing Low-Cost Molecular Property Prediction with Contrastive Learning on SMILES Representations

#### Authors: [Marcos G. Quiles](https://scholar.google.com.br/citations?user=kQXxkc4AAAAJ&hl=pt-BR&oi=ao), Piero A. L. Ribeiro, [Gabriel A. Pinheiro](https://scholar.google.com.br/citations?user=819H8Y8AAAAJ&hl=pt-BR&oi=ao), [Ronaldo C. Prati](https://scholar.google.com.br/citations?user=lZ0ASREAAAAJ&hl=pt-BR), [Juarez L. F. Da Silva](https://scholar.google.com.br/citations?user=wQG1X8wAAAAJ&hl=pt-BR&oi=ao)

#### [Link to Paper](https://link.springer.com/chapter/10.1007/978-3-031-65329-2_26)

The official PyTorch implementation of Enhancing Low-Cost Molecular Property Prediction with Contrastive Learning on SMILES Representations. This paper explores self-supervised contrastive learning techniques in the Simplified Molecular Input Line Entry System (SMILES) representations.

#### Usage

Sample command to run CL training via SMILES enumeration with ZINC dataset
```
$ python3 main.py --epochs=101 --no-lr-decay --temperature=.1 --batch=256 --output result_seed_12200 --bidirectional --embedding_dim=64 --num-layers=3 --lstm_dim=64 --seed 12200
```
Sample command to run the finetuning supervised training
```
$ python3 main.py --epochs=301 --lr=1e-3 --batch=32 --load_weights result_seed_12200 --output sup_result --bidirectional --embedding_dim=64 --num-layers=3 --lstm_dim=64 --sup --target 15 --seed 12200 --output sup_15_12200 --qm9
```

#### Cite

Please cite [our paper]([...](https://link.springer.com/chapter/10.1007/978-3-031-65329-2_26)) if you use this code in your own work:

```
@inproceedings{Quiles_2024_CL_SMILES,
  title={Enhancing Low-Cost Molecular Property Prediction with Contrastive Learning on SMILES Representations},
  author={Quiles, Marcos G. and Ribeiro, Piero A. L. and Pinheiro, Gabriel A. and Prati, Ronaldo C. and Silva, Juarez L. F. da},
  booktitle={International Conference on Computational Science and Its Applications},
  pages={387--401},
  year={2024},
  organization={Springer}
}
```
