"""

# @author: Yue Peng
# @email: yuepaang@gmail.com
# @createTime: 2018-10-31, 08:42:00 GMT+0800
# @description: Triplet Loss + Topic Memory Module
"""
import os
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import math
from torch.optim.optimizer import Optimizer
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm, trange
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


# ====================
# Model Part
# ====================
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training or self.stddev <= 0.0:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
            # return din + torch.autograd.Variable(torch.randn(din.size()) * self.stddev)
        return din


class Model(nn.Module):
    def __init__(self, emb, **kwargs):
        super(Model, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=kwargs['freeze_wv'])
        self.fc1 = nn.Linear(emb.shape[1], kwargs['hidden_dim'])
        self.lstm = nn.LSTM(
            kwargs['hidden_dim'],
            kwargs['hidden_dim'],
            num_layers=kwargs['num_layers'],
            bidirectional=False,
            batch_first=True)
        self.fc2 = torch.nn.utils.weight_norm(nn.Linear(kwargs['hidden_dim'], kwargs['num_classes']))
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.noise = GaussianNoise(kwargs['noise'])


        # Inference
        self.fc3 = nn.Linear(emb.shape[1], kwargs['num_classes'])
        self.fc4 = nn.Linear(kwargs['num_classes'], kwargs['num_classes'])
        self.relu = nn.ReLU()
        # mu
        self.fc51 = nn.Linear(kwargs['num_classes'], kwargs['num_classes'])
        # log_sigma
        self.fc52 = nn.Linear(kwargs['num_classes'], kwargs['num_classes'])
        # Generative
        self.fc6 = nn.Linear(kwargs['num_classes'], emb.shape[1])
        # Source 
        self.fc7 = nn.Linear(emb.shape[1], emb.shape[1])
        # Target
        self.fc8 = nn.Linear(emb.shape[1], emb.shape[1])

    def forward(self, x, x_lens):
        x_emb = self.emb(x)
        x_emb = self.noise(x_emb)
        b, l, e = x_emb.size()
        # sentence representation
        x_repre = torch.sum(x_emb, 1) # (B, E)
        # Neural Variational Inference
        lambda_ = self.relu(self.fc3(x_repre)) # (B, K)
        pi_ = self.relu(self.fc4(lambda_)) # (B, K)
        mu = self.fc51(pi_) # (B, K)
        log_sigma = self.fc52(pi_) # (B, K)
        sigma = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(sigma)
        # Q(h|X)
        h_ = eps.mul(sigma).add_(mu) # (B, K)
        # P(X|h)
        recon_x = self.fc6(h_) # (B, E)
        p_x_i = -1.0 * recon_x
        w_phi = self.fc6.weight.cuda() # (E, K)
        # Source Memory
        s = self.relu(self.fc7(w_phi.transpose(0, 1))) # (K, E)
        # Target Memory
        t = self.relu(self.fc8(w_phi.transpose(0, 1))) # (K, E)
        # Probability
        p = torch.mm(x_emb.view(-1, e), s.t())
        p = p.view(b, l, -1) # (B, L, K)
        p = torch.transpose(p, 1, 2) # (B, K, L)
        p = torch.sum(p, dim=2) # (B, K)
        # output representation
        zeta = 0.8 * p + h_ #(B, K)
        R = zeta.mm(t) # (B, E)
        R = R.view(R.size(0), 1, R.size(1)) # (B, 1, E)
        x_new = torch.cat((x_emb, R), 1) # (B, L+1, E)

        h = self.fc1(x_new)
        h = nn.utils.rnn.pack_padded_sequence(h, x_lens, batch_first=True)
        _, (h, c) = self.lstm(h, None)
        hidden = h[0]
        h = self.dropout(hidden)
        logits = self.fc2(h)
        return logits, hidden, x_repre, recon_x, mu, log_sigma


# =====================
# Dataset Pipeline
# =====================
def load_embedding(fn_emb):
    word_vectors = KeyedVectors.load_word2vec_format(fn_emb, binary=True)
    word2index = {}
    zeros = np.zeros(word_vectors.vectors.shape[1], dtype=np.float32)
    embedding = np.insert(word_vectors.vectors, 0, zeros, axis=0)
    print("emb:", embedding.shape)
    padding_value = 0
    for i, w in enumerate(word_vectors.index2word):
        word2index[w] = i + 1
    return embedding, word2index, padding_value


def to_index(text, word2index):
    text = text.lower()
    l = []
    for w in jieba.cut(text):
        if w.replace("%", "").replace(".", "").isdigit():
            w = "0"
        try:
            l.append(word2index[w])
        except KeyError:
            l.append(0)
    return l


class NlpDataSet(Dataset):
    def __init__(self, fn_data, word2index, padding_value, transforms=None):
        super(NlpDataSet, self).__init__()
        self.word2index = word2index
        self.padding_value = padding_value
        self.df = None
        self.labels = None
        self.data = None
        self.load_data(fn_data)
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = lambda x: x

    def load_data(self, fn_data):
        self.df = pd.read_csv(fn_data).reset_index(drop=True)
        self.labels = pd.unique(self.df['label'])
        data = []
        for _, row in self.df.iterrows():
            text = row['query']
            label = row['label']
            ll = to_index(text, self.word2index)
            data.append((ll, label))
        self.data = np.array(data)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        data = self.data[index]
        return self.transforms(data[0]), data[1]

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):

    def __init__(self, ds, p, k):
        self.ds = ds
        self.p = p
        self.k = k
        # from data-frame
        # unique labels
        self.labels = ds.labels
        self.samples = ds.df

    def __iter__(self):
        for i in range(self.samples.shape[0] // (self.p * self.k)):
            np.random.shuffle(self.labels)
            # Pick up 64 labels randomly
            labels = self.labels[:self.p]
            batch = []
            for label in labels:
                samples = self.samples[self.samples['label'] == label]
                if samples.shape[0] < 2:
                    continue
                if samples.shape[0] > self.k:
                    samples = samples.sample(self.k)
                # 8 same label sample
                batch += samples.index.tolist()
            # Not fix batch size sampler (64•[2~8])
            yield batch

    def __len__(self):
        return self.samples.shape[0] // (self.p * self.k)


class CollateFn(object):
    def __init__(self, max_len=None, padding_value=0):
        self.max_len = max_len
        self.padding_value = padding_value

    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        seqs, labels = zip(*batch)
        lengths = list(map(len, seqs))
        max_len = self.max_len if self.max_len is not None else len(seqs[0])

        padded = []
        for seq in seqs:
            if len(seq) < max_len:
                padded.append(seq + [self.padding_value] * (max_len - len(seq)))
            else:
                padded.append(seq[0:max_len])

        return padded, labels, lengths


# ========================
# Optimizer Part
# ========================
class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss


# ====================
# Triplet Loss
# ====================
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


# ===================
# Topic Network Loss
# ===================
def loss_function(recon_x, x_repre, mu, log_sigma_sq):
    """Return Loss

    Arguments:
        x {tensor} -- indices of vocabulary
        p_x_i {tensor} -- probability after softmax layer
        mu {[type]} -- [description]
        log_sigma_sq {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    LL = nn.functional.cosine_similarity(recon_x, x_repre, dim=1) # (B, 1)
    LL = -0.5 * torch.mean(LL)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())

    return LL + KLD


# ===================
# Training
# ===================
P = 64
K = 6
num_workers = 4

fn_emb = "/data/embedding_word_300_20180801.bin"
embedding, word2index, padding_value = load_embedding(fn_emb)

train_ds = NlpDataSet("/data/181016/data_sample_132.csv", word2index, padding_value)
valid_ds = NlpDataSet("/data/181016/poc_test_132.csv", word2index, padding_value)

train_dl = DataLoader(
    train_ds, num_workers=num_workers, batch_sampler=BatchSampler(train_ds, P, K), collate_fn=CollateFn())
valid_dl = DataLoader(
    valid_ds, num_workers=num_workers, batch_size=128, collate_fn=CollateFn())

args = {
    "num_layers": 1,
    "hidden_dim": 256,
    "num_classes": 132,
    "dropout": 0.5,
    "freeze_wv": True,
    "noise": 0.0,
}

device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# device = torch.device('cpu')
margin = 0.1
weight_decay = 5 * 1e-4
lr = 1e-2

model = Model(embedding, **args)
model.to(device)

triplet_loss_fn = TripletLoss(margin)
class_loss_fn = nn.CrossEntropyLoss()


def get_lr(opt):
    for pg in opt.param_groups:
        return pg['lr']


def calc_topk_num(logits, labels, k=3):
    topk = logits.topk(k)[1].tolist()
    labels = labels.tolist()
    c = 0
    for l, t in zip(labels, topk):
        if l in t:
            c += 1
    return c


writer = SummaryWriter("/data/log")


if __name__ == "__main__":
    with tqdm(total=10000) as t:
        def run(model, optimizer, dl, i, train=True, max_steps=None):
            if train:
                model.train()
                prefix = "train"
            else:
                model.eval()
                prefix = "valid"

            if max_steps is None:
                max_steps = len(dl)
            else:
                max_steps = min(max_steps, (len(dl)))
            num_right = 0
            num_total = 0
            top_k_right = 0
            total_loss = 0.0
            steps = 0

            for x, y, lens in dl:
                x = torch.tensor(x).to(device)
                y = torch.tensor(y).to(device)
                lens = torch.tensor(lens).to(device)
                logits, emb, x_repre, recon_x, mu, log_sigma = model(x, lens)

                if train:
                    loss1, ap, an = triplet_loss_fn(emb, y, normalize_feature=False)
                else:
                    loss1, ap, an = torch.tensor(0.0, device=device), 0, 0

                loss2 = class_loss_fn(logits, y)
                loss3 = loss_function(recon_x, x_repre, mu, log_sigma)
                loss = loss1 + loss2 + loss3
                if train:
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                total_loss += loss.item()
                steps += 1
                avg_loss = total_loss / float(steps)

                pred = logits.argmax(dim=-1)

                num_right += (pred == y).sum().item()

                if not train:
                    top_k_right += calc_topk_num(logits, y, k=3)

                num_total += x.size(0)
                acc = num_right / float(num_total)
                topk_acc = top_k_right / float(num_total)
                t.set_description('Iter %i' % i)
                t.set_postfix(
                    loss=loss.item(),
                    avg_loss=avg_loss,
                    loss1=loss1.item(),
                    loss2=loss2.item(),
                    loss3 = loss3.item(),
                    acc=acc,
                    topk=topk_acc,
                    steps=steps,
                    type=prefix,
                    lr=get_lr(optimizer),
                    max_steps=max_steps,
                    ap=0,
                    an=0)
                writer.add_scalar('%s/loss' % prefix, loss.item(), i * max_steps + steps)

                if steps == max_steps:
                    break

            t.update()
            writer.add_scalar('%s/avg_loss' % prefix, total_loss / float(steps), i)
            writer.add_scalar('%s/acc' % prefix, num_right / float(num_total), i)
            writer.add_scalar('%s/lr' % prefix, get_lr(optimizer), i)
            if not train:
                print("epoch:", i, "avg_loss:", total_loss / float(steps), "acc:", num_right / float(num_total), "topk:",
                      top_k_right / float(num_total))

        epochs = 0
        optimizer = AdamW(model.parameters(), lr=7e-4, weight_decay=weight_decay)
        # run(model,optimizer,valid_dl,epochs,train=False)
        for epochs in range(10000):
            run(model, optimizer, train_dl, epochs, train=True, max_steps=300)
            if epochs > 0 and epochs % 2 == 0:
                run(model, optimizer, valid_dl, epochs, train=False)
            if epochs > 0 and epochs % 10 == 0:
                torch.save(model.state_dict(), "/data/q2q/epoch-%s" % epochs)
