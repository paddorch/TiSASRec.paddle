import random
import copy
import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn.functional as F

from tisasrec.utils import computeRePos


class MyBCEWithLogitLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyBCEWithLogitLoss, self).__init__()

    def forward(self, pos_logits, neg_logits, labels):
        return paddle.sum(
            - paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels -
            paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels,
            axis=(0, 1)
        ) / paddle.sum(labels, axis=(0, 1))


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users, mininterval=5):

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(
            *[paddle.to_tensor(l, dtype="int64") for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users, mininterval=5):
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(
            *[paddle.to_tensor(l, dtype="int64") for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user
