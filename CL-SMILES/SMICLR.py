import math
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class SMILESEncoder(torch.nn.Module):

    def __init__(
        self, vocab_size, max_len, padding_idx, embedding_dim=64,
        dim=128, num_layers=1, bidirectional=False
    ):
        super(SMILESEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers

        self.encoder = torch.nn.LSTM(
            self.embedding_dim,
            self.dim,
            self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
    def forward(self, x, len):

        x = rnn_utils.pack_padded_sequence(x, len, batch_first=True, enforce_sorted=False)
        
        feat, (hidden, _) = self.encoder(x)
        if self.bidirectional:
            return torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
        return hidden[-1]


class Net(torch.nn.Module):
    def __init__(self, dim, vocab_size, max_len, padding_idx,
                 embedding_dim=64, num_layers=1, bidirectional=False):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers

        self.emb = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        # Base encoders
        self.SMILESEnc1 = SMILESEncoder(
            self.vocab_size, self.max_len, self.padding_idx,
            self.embedding_dim, 2 * self.dim, 
            self.num_layers, self.bidirectional
        )

        self.SMILESEnc2 = SMILESEncoder(
            self.vocab_size, self.max_len, self.padding_idx,
            self.embedding_dim, 2 * self.dim, 
            self.num_layers, self.bidirectional
        )

        # Projection head
        self.g = torch.nn.Sequential(
            torch.nn.Linear((4 if bidirectional else 2) * self.dim, 4 * self.dim),
            torch.nn.BatchNorm1d(4 * self.dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * self.dim, 6 * self.dim, bias=False)
        )

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.dim
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, smi, random_smi, smi_lengths, random_smi_length):
        x = smi.view(-1, self.max_len)
        x = self.emb(x)

        x1 = random_smi.view(-1, self.max_len)
        x1 = self.emb(x1)

        enc1 = self.SMILESEnc1(x, smi_lengths)
        enc2 = self.SMILESEnc2(x1, random_smi_length)
        return F.normalize(self.g(enc1), dim=1), F.normalize(self.g(enc2), dim=1)


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):

    out = torch.cat([out_1, out_2], dim=0)

    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(
        math.e**(1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = torch.log(pos / (neg + eps))
    return -loss.mean(), loss.size(0)


def test(model, loader, device, temperature):

    model.eval()
    error = 0
    total_num = 0

    for data in loader:
        smi_lengths = data.random_smi_len1
        random_smi_lengths = data.random_smi_len2
        data = data.to(device)
        out_1, out_2 = model(data.random_smi1, data.random_smi2, smi_lengths, random_smi_lengths)
        loss, batch_size = nt_xent_loss(out_1, out_2, temperature)
        error += loss.item() * batch_size
        total_num += batch_size

    return error / total_num


def train(model, unsup_loader, optimizer, device, output, temperature):
    model.train()
    loss_all = 0
    total_num = 0

    for unsup in unsup_loader:
        smi_lengths = unsup.random_smi_len1
        random_smi_lengths = unsup.random_smi_len2
        unsup = unsup.to(device)
        optimizer.zero_grad()
        out_1, out_2 = model(unsup.random_smi1, unsup.random_smi2, smi_lengths, random_smi_lengths)
        loss, batch_size = nt_xent_loss(out_1, out_2, temperature)
        loss.backward()
        loss_all += loss.item() * batch_size
        total_num += batch_size
        optimizer.step()

    return loss_all / total_num
