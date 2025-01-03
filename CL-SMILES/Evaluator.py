import torch
import torch.nn.functional as F


from SMICLR import SMILESEncoder


class Net(torch.nn.Module):
    def __init__(self, dim, vocab_size, max_len, padding_idx,
                embedding_dim=64, num_layers=1, bidirectional=False, freeze_encoder=False):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers
        self.freeze_encoder = freeze_encoder

        self.emb = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        self.SMILESEnc1 = SMILESEncoder(
            self.vocab_size, self.max_len, self.padding_idx,
            self.embedding_dim, 2 * self.dim, 
            self.num_layers, self.bidirectional
        )

        # Freeze SMILESEncoder weights if freeze_encoder is True
        if self.freeze_encoder:
            for param in self.SMILESEnc1.parameters():
                param.requires_grad = False

        self.fc1 = torch.nn.Linear((4 if bidirectional else 2) * self.dim, self.dim)
        self.fc2 = torch.nn.Linear(self.dim, 1)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.dim
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, smi, len):
        x = smi.view(-1, self.max_len)
        x = self.emb(x)

        out = self.SMILESEnc1(x, len)
        out = F.tanh(self.fc1(out))
        out = self.fc2(out)
        return out.view(-1)


def test(model, loader, device, std):
    model.eval()
    error_smi = 0
    error_random = 0
    for data in loader:
        lengths = data.smi_len
        smi_lengths = data.random_smi_len1
        data = data.to(device)
        pred1 = model(data.smi, lengths)
        pred2 = model(data.random_smi1, smi_lengths)
        error_smi += (pred1 * std - data.y * std).abs().sum().item()  # MAE
        error_random += (pred2 * std - data.y * std).abs().sum().item()  # MAE
    return (error_smi + error_random) / (len(loader.dataset) * 2), error_smi / len(loader.dataset), error_random / len(loader.dataset)


def train(model, loader, optimizer, device, output, std=None):
    model.train()
    loss_all = 0
    for data in loader:
        smi_lengths = data.smi_len
        random_smi_lengths = data.random_smi_len1
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data.smi, smi_lengths), data.y)
        loss += F.mse_loss(model(data.random_smi1, random_smi_lengths), data.y)
        loss.backward()
        loss_all += loss.item() * data.y.size(0) * 2
        optimizer.step()
    return loss_all / (len(loader.dataset) * 2)
