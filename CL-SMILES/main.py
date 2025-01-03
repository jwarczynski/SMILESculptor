# ===================================================
# IMPORTING PACKAGES
# ===================================================
import numpy as np
import random
import torch
import torch_geometric.transforms as T
import time
import os
import matplotlib.pyplot as plt


from arguments import arg_parse
from datetime import datetime
from datasets import *
from transform import Complete
from pathlib import Path
from os.path import basename
from torch_geometric.loader import DataLoader

# ===================================================
# BASIC FUNCTIONS
# ===================================================

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


if __name__ == '__main__':
    
    args = arg_parse()
    seed = args.seed
    seed_everything(seed)
    max_len = 36
    char_indices = Complete().get_char_indices()
    nchars = len(char_indices)

    target = args.target
    epochs = args.epochs
    sup = args.sup
    embedding_dim = args.embedding_dim
    lr_decay = args.lr_decay
    lr = args.lr
    lstm_dim = args.lstm_dim
    weight_decay = args.weight_decay
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    temperature = args.temperature
    batch_size = args.batch_size
    sup_size = args.sup_size
    output = args.output
    load_weights = args.load_weights
    freeze_encoder = args.freeze_encoder

    data_qm9 = args.data_qm9
    data_anions = args.data_anions
    data_cations = args.data_cations
    lumo = args.lumo
    print(data_qm9, data_anions, data_cations, lumo)
    if not Path(output).exists():
        Path(output).mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Using {} for training ...'.format(device))
    
    if sup:
        from Evaluator import *

        if data_qm9:
            dataset = QM9(
                'qm9',
                transform=T.Compose(
                    [MyTransform(), Complete(augmentation=True, max_len=max_len)]
                )
            )[np.load('qm9_seed_{}.npy'.format(seed))]
            train_lims = 20000
            val_lims = 10000

        train_dataset = dataset[train_lims:train_lims + sup_size]
            
        mean = train_dataset.data.y.mean(dim=0, keepdim=True)
        std = train_dataset.data.y.std(dim=0, keepdim=True)
        dataset._data.y = (dataset._data.y - mean) / std

        if data_qm9:
            mean, std = mean[:, target].item(), std[:, target].item()
        
        test_dataset = dataset[val_lims:train_lims]
        val_dataset = dataset[:val_lims]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        temperature = None
    else:
        from SMICLR import *

        zinc = ZINC('zinc').shuffle()
        qm9 = QM9('qm9').shuffle()

        np.save('zinc_seed_{}'.format(seed), np.array(zinc.indices()))
        np.save('qm9_seed_{}'.format(seed), np.array(qm9.indices()))

        # merging Datasets
        zinc = [data for data in zinc]
        qm9_v2 = []
        
        for data in qm9[20000:]:
            del data.y
            qm9_v2.append(data)

        dataset = zinc[100000:] + qm9_v2
        random.shuffle(dataset)
        train_dataset = UnlabeledDataset(
            'unlabeled_training', 
            dataset, transform=Complete(augmentation=True, max_len=max_len)
        ).shuffle()

        qm9_v2 = []
        
        for data in qm9[:10000]:
            del data.y
            qm9_v2.append(data)

        dataset = zinc[:100000] + qm9_v2
        random.shuffle(dataset)
        val_dataset = UnlabeledDataset(
            'unlabeled_val', 
            dataset, transform=Complete(augmentation=True, max_len=max_len)
        ).shuffle()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net(
        lstm_dim, vocab_size=nchars, max_len=max_len,
        padding_idx=char_indices[''], embedding_dim=embedding_dim,
        num_layers=num_layers, bidirectional=bidirectional, freeze_encoder=freeze_encoder
    ).to(device)
    
    if load_weights:
        print(f"loading weights from {load_weights}")
        weights = torch.load('{}/best_unsup_model.pth'.format(load_weights))
        model_dict = model.state_dict()

        weights_dict = {k: v for k, v in weights.items() if k in model_dict}
        model_dict.update(weights_dict)
        model.load_state_dict(model_dict)
           
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10,min_lr=0.00001
        )
    log_file = "{}/{}_history.log".format(output, 'sup' if sup else 'unsup')

    val_loss = []
    train_loss = []
    best_val_error = None
    for epoch in range(1, epochs):
    
        if lr_decay:
            lr = scheduler.optimizer.param_groups[0]['lr']
        init = time.time()
        loss = train(model, train_loader, optimizer, device, output, temperature)
        final = time.time() - init  
        
        if sup:
            val_error, val_original, val_random  = test(model, val_loader, device, std)
            val_loss.append(val_error)
            train_loss.append(test(model, train_loader, device, std)[0])

            if best_val_error is None or val_error <= best_val_error:
                torch.save(model.state_dict(), '{}/best_model.pth'.format(output))
                test_error, test_original, test_random  = test(model, test_loader, device, std)
                best_val_error = val_error

            msg = (
                'Epoch: {:03d}, LR: {:7f}, Loss (Norm/Real): {:.7f}/{:.7f}, Validation Loss (total/original/random): {:.7f}, {:.7f}, {:.7f} Test Loss (total/original/random): {:.7f}, {:.7f}, {:.7f},  time: {}'.format(
                    epoch, lr, loss, train_loss[-1], val_error, val_original, val_random, 
                    test_error, test_original, test_random, time.strftime("%H:%M:%S", time.gmtime(final))
                )
            )
            
        else:
            val_error = test(model, val_loader, device, temperature)
            val_loss.append(val_error)

            if best_val_error is None or val_error <= best_val_error:
                torch.save(model.state_dict(), '{}/best_unsup_model.pth'.format(output))
                best_val_error = val_error

            train_loss.append(loss)
            msg = (
                'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation Loss: {:.7f}, time: {}'.format(
                epoch, lr, loss, val_error, time.strftime("%H:%M:%S", time.gmtime(final)))
            )

        if lr_decay:
            scheduler.step(val_error)

        with open(log_file, 'a') as f:
            f.write(msg+ '\n')
      
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('{}/{}_loss_per_epochs.png'.format(output, 'sup' if sup else 'unsup'))
