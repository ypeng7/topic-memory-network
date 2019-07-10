"""
TODO:
# @author: Yue Peng
# @email: yuepaang@gmail.com
# @createTime: 2018-11-02, 20:49:48 GMT+0800
# @description: Char Level CNN Classifier with Attention
"""
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import load_embedding, OnlineQA
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Character Level CNN with Attention')
# # Model options
# parser.add_argument('--depth', default=16, type=int)
# parser.add_argument('--width', default=1, type=float)
# parser.add_argument('--dataset', default='CIFAR10', type=str)
# parser.add_argument('--dataroot', default='.', type=str)
# parser.add_argument('--dtype', default='float', type=str)
# parser.add_argument('--nthread', default=4, type=int)
# parser.add_argument('--teacher_id', default='', type=str)

# # Training options
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.1, type=float)
# parser.add_argument('--epochs', default=200, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
# parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# parser.add_argument('--resume', default='', type=str)
# parser.add_argument('--randomcrop_pad', default=4, type=float)
# parser.add_argument('--temperature', default=4, type=float)
# parser.add_argument('--alpha', default=0, type=float)
# parser.add_argument('--beta', default=0, type=float)

# # Device options
# parser.add_argument('--cuda', action='store_true')
# parser.add_argument('--save', default='', type=str,
#                     help='save parameters and logs in this folder')
# parser.add_argument('--ngpu', default=1, type=int,
#                     help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')


class CharCNN(nn.Module):
	def __init__(self, emb, args):
		super(CharCNN, self).__init__()
		# Embedding
		self.embedding_size = emb.shape[1]
		self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True)
		self.emb_dropout = nn.Dropout(args.emb_dropout)
		# CNN
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, args.num_filters, (2, self.embedding_size), 1),
			nn.ReLU()
			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(1, args.num_filters, (3, self.embedding_size), 1),
			nn.ReLU()
			)
		self.conv3 = nn.Sequential(
			nn.Conv2d(1, args.num_filters, (4, self.embedding_size), 1),
			nn.ReLU()
			)
		# fc
		self.dropout = nn.Dropout(args.dropout)
		self.fc = nn.Linear(3*args.num_filters, args.num_classes)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)
        init.xavier_normal_(self.conv3.weight)
        init.xavier_normal_(self.fc.weight)

	def forward(self, x):
		x = self.emb(x)
		x = self.emb_dropout(x) # (B, L, E)
		x = x.unsqueeze(1) # (B, 1, L, E)
		# (B, f, L-*, 1)
		c1 = self.conv1(x)
		c2 = self.conv2(x)
		c3 = self.conv3(x)
		# (B, f, L-*)
		c1 = c1.squeeze(3)
		c2 = c2.squeeze(3)
		c3 = c3.squeeze(3)
		# (B, f, 1)
		h1 = nn.MaxPool1d(c1.size(2), 1)(c1)
		h2 = nn.MaxPool1d(c2.size(2), 1)(c2)
		h3 = nn.MaxPool1d(c3.size(2), 1)(c3)
		# (B, 3f)
		output = torch.cat((h1.squeeze(2), h2.squeeze(2), h3.squeeze(2)), 1)
		output = self.dropout(output)
		logits = self.fc(output)

		return logits


class args:
	emb_dropout = 0.1
	num_filters = 64
	num_classes = 638
	dropout = 0.1

	learning_rate = 1e-2
	log_interval = 300


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda")

embedding, char2index, padding_value = load_embedding()

# train_fn = "181016/data_sample_132.csv"
# test_fn = "181016/poc_test_132.csv"
train_fn = "fengbot/train.csv"
test_fn = "fengbot/valid.csv"

train_dataset = OnlineQA(max_len=40, data_fn=train_fn, char2index=char2index)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)

test_dataset = OnlineQA(max_len=40, data_fn=test_fn, char2index=char2index)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=False)


model = CharCNN(embedding, args)
model.to(device)


parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameter, lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# os.rmdir("/data/log/fengbot1")
os.mkdir("/data/log/fengbot1")
train_writer = SummaryWriter("/data/log/fengbot1")
test_writer = SummaryWriter("/data/log/fengbot1")


def train(epoch):
    model.train()
    train_loss = 0
    num_correct = 0
    num_total = 0
    for batch_idx, (_, x_seq, y) in enumerate(train_loader):
        x_seq = x_seq.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x_seq)
        # prediction
        pred = logits.argmax(-1)
        num_correct += (pred == y).sum().item()
        num_total += x_seq.size(0)
        # loss
        loss = criterion(logits, y)
        # # L2-Loss
        # lambd = torch.tensor(1.).cuda()
        # l2_reg = torch.tensor(0.).cuda()
        # for param in model.parameters():
        # 	l2_reg += torch.norm(param)
        # loss += lambd * l2_reg
        loss = loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_seq),
                len(train_dataset.df),
                100. * batch_idx / (len(train_dataset.df) // 64),
                loss.item() / len(x_seq))
                )

    print('=================> Epoch: {} Average loss: {:.4f} Train Set Accuracy: {:.4f}'.format(
        epoch,
        train_loss / len(train_dataset.df),
        num_correct / float(num_total))
        )
    train_writer.add_scalar("train-loss", train_loss / len(train_dataset.df), epoch)
    train_writer.add_scalar("train-acc", num_correct / float(num_total), epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for i, (_, x_seq, y) in enumerate(test_loader):
            x_seq = x_seq.to(device)
            y = y.to(device)
            logits = model(x_seq)
            # prediction
            pred = logits.argmax(-1)
            num_correct += (pred == y).sum().item()
            num_total += x_seq.size(0)
            # loss
            loss = criterion(logits, y)
            test_loss += loss.item()

    global test_acc
    test_acc = num_correct / float(num_total)

    print('=================> Test set accuracy: {:.4f}'.format(test_acc))

    test_writer.add_scalar("valid-loss", test_loss / len(test_dataset.df), epoch)
    test_writer.add_scalar("valid-acc", num_correct / float(num_total), epoch)


def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)


def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr, momentum=0.9, weight_decay=opt.weight_decay)


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    optimizer = create_optimizer(opt, opt.lr)

    for epoch in range(1, 10000):
        train(epoch)
        test(epoch)
        if test_acc > 0.827:
            torch.save(model.state_dict(), "/data/tmn/epoch-{}-{:.4f}".format(epoch, test_acc))


if __name__ == '__main__':
    main()
