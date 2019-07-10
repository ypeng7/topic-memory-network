# -*- coding:utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
# Created On 2018.10.30
import os
import sys
import torch
from torch.utils.data import DataLoader
from data_loader import load_embedding, OnlineQA
from tensorboardX import SummaryWriter
# ==========================
#    TMN GPU Version
# ==========================
class InferenceEncoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, args):
        super(InferenceEncoder, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Inference (Using Bag of Word representation)
        self.fc1 = torch.nn.Linear(self.vocab_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(args.dropout)
        # mu
        self.fc31 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # log_sigma
        self.fc32 = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        h1 = self.fc1(x)
        h1 = self.dropout(h1)
        h1 = self.relu(h1)
        h2 = self.fc2(h1)
        h2 = self.relu(h2)
        mu = self.fc31(h2)
        log_sigma_sq = self.fc32(h2)

        return mu, log_sigma_sq


class GenerativeDecoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, args):
        super(GenerativeDecoder, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        # Generative
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, self.vocab_size)
        # Topic Memory
        # S
        self.fc51 = torch.nn.Linear(self.vocab_size, self.embedding_size)
        # T
        self.fc52 = torch.nn.Linear(self.vocab_size, self.embedding_size)

    def forward(self, h):
        e = -1.0 * self.fc4(x)
        p_x_i = torch.nn.functional.softmax(e, dim=1)
        w_phi = self.fc4.weight
        w_phi = w_phi.cuda()
        s = self.fc51(torch.transpose(w_phi, 1, 0))
        s = self.relu(s)

        t = self.fc52(torch.transpose(w_phi, 1, 0))
        t = self.relu(t)
        return p_x_i, s, t


class TMN(torch.nn.Module):
    """

    Arguments:
        torch {[type]} -- [description]
    """

    def __init__(self, emb, vocab_size, hidden_size, embedding_size, args):
        """[summary]

        Arguments:
            vocab_size {[type]} -- V
            hidden_size {[type]} -- K
            args {[type]} --
        """

        super(TMN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.inference_encoder = InferenceEncoder(self.vocab_size, self.hidden_size, self.args)

        self.generative_decoder = GenerativeDecoder(self.vocab_size, self.hidden_size, self.embedding_size, self.args)

        # Embedding
        self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False)
        self.emb_dropout = torch.nn.Dropout(0.1)
        # CNN
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv1d(self.embedding_size, self.embedding_size, 1, 1),
        #     torch.nn.ReLU()
        # )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_size, self.embedding_size, 2, 1),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_size, self.embedding_size, 3, 1),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_size, self.embedding_size, 4, 1),
            torch.nn.ReLU()
        )
        # fc
        self.fc6 = torch.nn.Linear(2*self.embedding_size, self.hidden_size)

    def forward(self, x_oh, x_seq):
        # embedding
        x_emb = self.emb(x_seq)
        x_emb = self.emb_dropout(x_emb) # (b, L, E)
        # CNN
        # (b, E, L) => (b, E)
        c = self.cnn_clf(x_emb.transpose(1, 2))
        # inference
        x_bow = x_oh # (b, V)
        # mu, log_sigma_sq = self.inference_model(x_bow)
        mu, log_sigma_sq = self.inference_encoder(x_bow)
        # parameterize
        h = self.reparameterize(mu, log_sigma_sq)

        p_x_i, s, t = self.generative_decoder(h)

        p = self.match_prob(s, x_emb)
        zeta = self.integrated_memory_weight(p, h, gamma=0.8)
        R = self.output_representation(zeta, t) #(B, E)
        x_new = torch.cat((c, R), 1) # (b, 2E)
        x_new = self.dropout(x_new)
        logits = self.fc6(x_new)

        return logits, p_x_i, mu, log_sigma_sq

    def reparameterize(self, mu, log_sigma_sq):
        sigma = torch.exp(0.5*log_sigma_sq)
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def match_prob(self, s, x_emb):
        """Compute the match between
        the k-th topic and the embedding of the l-th word

        Arguments:
            s {[type]} -- (K, E)
            u {[type]} -- (b, L, E)

        Returns:
            [type] -- (b, k)
        """
        b, l, e = x_emb.size()
        p = torch.mm(x_emb.view(-1, e), s.t())
        p = p.view(b, l, -1)
        p = torch.transpose(p, 1, 2)
        p = torch.sum(p, dim=2)

        return p

    def integrated_memory_weight(self, p, h, gamma=0.8):
        P = gamma * p
        zeta = P + h
        return zeta

    def output_representation(self, zeta, t):
        """Output representation of the topic memory mechanism

        Arguments:
            zeta {[type]} -- (b, K)
            t {[type]} -- (K, E)

        Returns:
            [type] -- (b, E)
        """

        R = zeta.mm(t)
        return R

    def cnn_clf(self, x):
        """CNN Classifier

        [description]

        Args:
            x: (B, E, L)

        Returns:
            [description]
            [type]
        """
        # 2-gram
        c2 = self.conv2(x) # (b, 300, n2)

        # 3-gram
        c3 = self.conv3(x) # (b, 300, n3)

        # 4-gram
        c4 = self.conv4(x) # (b, 300, n4)

        cnn_features = torch.cat([c2, c3, c4], 2) # (b, 300, n1+n2+n3)
        num_features = cnn_features.size(2)
        cnn_features = torch.nn.MaxPool1d(num_features, 1)(cnn_features)
        cnn_features = self.dropout(cnn_features) # (B, E, 1)
        cnn_features = cnn_features.view(-1, self.embedding_size)

        return cnn_features


def loss_function(p_x_i, x_seq, mu, log_sigma_sq):
    """Return Loss

    Arguments:
        x {tensor} -- indices of vocabulary
        p_x_i {(B, V)} -- probability after softmax layer
        mu {[type]} -- [description]
        log_sigma_sq {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    x = x_seq.long().cuda()
    mu = mu.cuda()
    log_sigma_sq = log_sigma_sq.cuda()
    # Corresponding Probability Log Likelihood
    LL = -1.0 * torch.sum(torch.log(torch.gather(p_x_i, 1, x) + 1e-10))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Kullback Leibler divergence
    KLD = -0.5 * torch.sum(1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())

    return LL + KLD


class args:
    dropout = 0.1
    log_interval = 200


embedding, char2index, padding_value = load_embedding()

train_fn = "data_sample_132.csv"
test_fn = "poc_test_132.csv"

train_dataset = OnlineQA(max_len=40, data_fn=train_fn, char2index=char2index)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)

test_dataset = OnlineQA(max_len=40, data_fn=test_fn, char2index=char2index)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=False)

model = TMN(embedding, vocab_size=len(char2index), hidden_size=132, embedding_size=300, args=args)


for p in model.parameters():
    p.requires_grad = True


# optim_g = torch.optim.Adam(model.generative_decoder.parameters(), lr=1e-3)
# optim_e = torch.optim.Adam(model.inference_encoder.parameters(), lr=1e-3)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

criterion = torch.nn.CrossEntropyLoss()

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda")

model.to(device)

writer = SummaryWriter("/data/log/tmn")


def calc_topk_num(logits, labels, k=3):
    topk = logits.topk(k)[1].tolist()
    labels = labels.tolist()
    c = 0
    for l, t in zip(labels, topk):
        if l in t:
            c += 1
    return c


def train(epoch):
    model.train()
    train_loss = 0
    num_correct = 0
    num_total = 0
    for batch_idx, (x_oh, x_seq, y) in enumerate(train_loader):
        x_oh = x_oh.to(device)
        x_seq = x_seq.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits, recon_batch, mu, logvar = model(x_oh, x_seq)
        # prediction
        pred = logits.argmax(-1)
        num_correct += (pred == y).sum().item()
        num_total += x_oh.size(0)
        # loss
        loss = 0.5 * loss_function(recon_batch, x_seq, mu, logvar) + criterion(logits, y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_seq),
                len(train_dataset.df),
                100. * batch_idx / (len(train_dataset.df) // 64),
                loss.item() / len(x_oh))
                )

    print('====> Epoch: {} Average loss: {:.4f} Train Set Accuracy: {:.4f}'.format(
        epoch,
        train_loss / len(train_dataset.df),
        num_correct / float(num_total))
        )
    writer.add_scalar("train-loss", train_loss / len(train_dataset.df), epoch)
    writer.add_scalar("train-acc", num_correct / float(num_total), epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    num_correct = 0
    num_total = 0
    top3_correct = 0
    with torch.no_grad():
        for i, (x_oh, x_seq, y) in enumerate(test_loader):
            x_oh = x_oh.to(device)
            x_seq = x_seq.to(device)
            y = y.to(device)
            logits, recon_batch, mu, logvar = model(x_oh, x_seq)
            # prediction
            pred = logits.argmax(-1)
            num_correct += (pred == y).sum().item()
            top3_correct += calc_topk_num(logits, y, k=3)
            num_total += x_seq.size(0)
            # loss
            loss = loss_function(recon_batch, x_seq, mu, logvar) + criterion(logits, y)
            test_loss += loss.item()

    test_acc = num_correct / float(num_total)
    if test_acc > 0.86:
        torch.save(model.state_dict(), "/data/tmn/epoch-%s-%.5f" % (epoch, test_acc))
    print('====> Test set accuracy: {:.4f} =============> Top 3 accuracy: {:.4f}'.format(num_correct / float(num_total), top3_correct/float(num_total)))

    writer.add_scalar("valid-loss", test_loss / len(test_dataset.df), epoch)
    writer.add_scalar("valid-acc", num_correct / float(num_total), epoch)


def main():
    for epoch in range(1, 1000):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    main()
