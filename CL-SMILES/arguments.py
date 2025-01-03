import argparse


def arg_parse():

    parser = argparse.ArgumentParser(
        description='SMICLR Contrastive Framework.'
    )

    parser.add_argument(
        '--target',
        dest='target',
        type=int,
        default=0,
        help='Target property'
    )
    
    parser.add_argument(
        '--lstm_dim',
        dest='lstm_dim',
        type=int,
        default=64,
        help='LSTM Encoder dimension'
    )
    
    parser.add_argument(
        '--qm9',
        dest='data_qm9',
        action='store_true',
    )
    
    parser.add_argument(
        '--cations',
        dest='data_cations',
        action='store_true',
    )
    
    parser.add_argument(
        '--anions',
        dest='data_anions',
        action='store_true',
    )
    
    parser.add_argument(
        '--lumo',
        dest='lumo',
        action='store_true',
    )
    
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=101,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--no-lr-decay',
        dest='lr_decay',
        action='store_false',
        help='Disable learning decay'
    )
    
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-6,
        help='Weight decay'
    )

    parser.add_argument(
        '--bidirectional',
        dest='bidirectional',
        action='store_true',
        help='Bidirectional RNN layer'
    )
    
    parser.add_argument(
        '--num-layers',
        dest='num_layers',
        type=int,
        default=1,
        help='Number of RNN layers'
    )

    parser.add_argument(
        '--temperature',
        dest='temperature',
        type=float,
        default=.1,
        help='Temperature'
    )
    
    parser.add_argument(
        '--batch',
        dest='batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    
    parser.add_argument(
        '--sup',
        dest='sup',
        action='store_true',
        help='Supervised training'
    )
    
    parser.add_argument(
        '--embedding_dim',
        dest='embedding_dim', 
        type=int,
        default=32,
        help='embedding dimension for SMILES'
    )

    parser.add_argument(
        '--sup-train-size',
        dest='sup_size',
        type=int,
        default=5000,
        help='Size of the supervised set'
    )
    
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        type=str,
        default='data',
        help='Folder to save the QM9 dataset'
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        default=1234,
        type=int
    )

    parser.add_argument(
        '--output',
        dest='output',
        type=str,
        default='exp',
        help='Output folder'
    )
    
    parser.add_argument(
        '--load-weights',
        dest='load_weights', 
        type=str,
        help='Path to load the weights of the model'
    )

    parser.add_argument(
        '--freeze-encoder',
        dest='freeze_encoder',
        action='store_true',
        help='Freeze the encoder weights'
    )

    return parser.parse_args()
