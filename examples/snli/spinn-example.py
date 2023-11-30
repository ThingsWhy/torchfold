"""This is just example model, not real implementation !!!"""

import time
import argparse
import sys

import torch

import torch.nn as nn
from torch import optim

from torchtext import data
from torchtext import datasets
from torchtext.data import Example

import torchfold


parser = argparse.ArgumentParser(description='SPINN')
parser.add_argument('--fold', action='store_true', default=False)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
args, _ = parser.parse_known_args(sys.argv)
args.cuda = not args.no_cuda and torch.cuda.is_available()


class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.left = nn.Linear(num_units, 5 * num_units)
        self.right = nn.Linear(num_units, 5 * num_units)

    def forward(self, left_in, right_in):
        lstm_in = self.left(left_in[0])
        lstm_in += self.right(right_in[0])
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        return h, c


class SPINN(nn.Module):

    def __init__(self, n_classes, size, n_words):
        super(SPINN, self).__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.embeddings = nn.Embedding(n_words, size)
        self.out = nn.Linear(size, n_classes)
        self.n_words=n_words

    def leaf(self, word_id):

        try:
            embedded = self.embeddings(word_id)
        except:
            print("Word {} is larger than n_words {}.".format(word_id, self.n_words))
        return embedded, \
               torch.zeros((word_id.size(0),
                            self.size))\
                   .type(dtype=torch.float32)

    def children(self, left_h, left_c, right_h, right_c):
        return self.tree_lstm((left_h, left_c), (right_h, right_c))

    def logits(self, encoding):
        return self.out(encoding)


def encode_tree_regular(model, tree):
    def encode_node(node):
        if node.is_leaf():
            return model.leaf(torch.Tensor([node.id],dtype=torch.long))
        else:
            left_h, left_c = encode_node(node.left)
            right_h, right_c = encode_node(node.right)
            return model.children(left_h, left_c, right_h, right_c)
    encoding, _ = encode_node(tree.root)
    return model.logits(encoding)


def encode_tree_fold(fold, tree):
    def encode_node(node):
        if node.is_leaf():
            return fold.add('leaf', node.id).split(2)
        else:
            left_h, left_c = encode_node(node.left)
            right_h, right_c = encode_node(node.right)
            return fold.add('children', left_h, left_c, right_h, right_c).split(2)
    encoding, _ = encode_node(tree.root)
    return fold.add('logits', encoding)


class Tree(object):
    class Node(object):
        def __init__(self, leaf=None, left=None, right=None):
            self.id = leaf
            self.left = left
            self.right = right

        def is_leaf(self):
            return self.id is not None

        def __repr__(self):
            return str(self.id) if self.is_leaf() else "(%s, %s)" % (self.left, self.right)

    def __init__(self, example, inputs_vocab, answer_vocab):
        self.label = answer_vocab.stoi[example.label] - 1
        queue = []
        idx, transition_idx = 0, 0
        while transition_idx < len(example.premise_transitions):
            t = example.premise_transitions[transition_idx]
            transition_idx += 1
            if t == 'shift':
                queue.append(Tree.Node(leaf=inputs_vocab.stoi[example.premise[idx]]))
                idx += 1
            else:
                n_left = queue.pop()
                n_right = queue.pop()
                queue.append(Tree.Node(left=n_left, right=n_right))
        assert len(queue) == 1
        self.root = queue[0]


def main():
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)
    print("Initializing dataset..")
    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)
    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=0 if args.cuda else -1)
    print("Done.")
    model = SPINN(3, 500, len(inputs.vocab))
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda' if args.cuda else 'cpu')

    for epoch in range(10):
        print("starting epoch {}".format(epoch))
        start = time.time()
        iteration = 0

        for batch_idx, batch in enumerate(train_iter):
            opt.zero_grad()

            all_logits, all_labels = [], []
            fold = torchfold.Fold(device=device)
            count=0
            # TODO: incorrect logic here, the for loop goes through the entire dataset, not the batch:
            for example in batch.dataset:
                #print("example:"+str(example))
                tree = Tree(example, inputs.vocab, answers.vocab)
                if args.fold:
                    all_logits.append(encode_tree_fold(fold, tree))
                else:
                    all_logits.append(encode_tree_regular(model, tree))
                all_labels.append(tree.label)
                # We use the following BAD workaround to test the later part of folding and training:
                count+=1
                if count>args.batch_size:
                    # we stop after seeing batchsize examples, but they are not from the batch, but from the
                    # entire dataset, presumably never shuffled.
                    break

            if args.fold:
                res = fold.apply(model, [all_logits, all_labels])
                loss = criterion(res[0], res[1])

            else:
                loss = criterion(torch.cat(all_logits, 0), torch.Tensor(all_labels,dtype=torch.long))
            print("loss={}".format(loss.item()))
            loss.backward(); opt.step()

            iteration += 1
            if iteration % 10 == 1:
                print("Avg. Time: %fs" % ((time.time() - start) / iteration))
                print("iteration {} loss:{}".format(iteration, loss))
                # iteration = 0
                start = time.time()


        print("done epoch {}".format(epoch))


if __name__ == "__main__":
    main()
