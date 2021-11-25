import os
import sys
import time
import paddle
import paddle.nn as nn
import pickle
import argparse

from tisasrec.model import TiSASRec
from tisasrec.train import *
from tisasrec.utils import *

parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)

# model
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--time_span', default=2048, type=int)

# train
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='gpu', type=str)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--eval_freq', default=20, type=int)
parser.add_argument('--log_freq', default=10, type=int)

set_seed(42)
args = parser.parse_args()

# gpu
if args.device:
    paddle.set_device(f"{args.device}")
print(paddle.get_device())

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
num_batch = len(user_train) // args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

try:
    relation_matrix = pickle.load(
        open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
except:
    relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
    pickle.dump(relation_matrix,
                open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))


sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen,
                      n_workers=3)

# paddle.nn.initializer.set_global_initializer(nn.initializer.XavierUniform())  # Do not use this!
model = TiSASRec(usernum, itemnum, itemnum, args)

model.train()  # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    checkpoint = paddle.load(args.state_dict_path)
    model.set_state_dict(checkpoint['state_dict'])
    epoch_start_idx = checkpoint['epoch'] + 1

if args.inference_only:
    model.eval()
    with paddle.no_grad():
        t_test = evaluate(model, dataset, args)
    print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

bce_criterion = MyBCEWithLogitLoss()
adam_optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, beta1=0.9, beta2=0.98)

T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only:
        break  # just to decrease identition
    for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()  # tuples to ndarray
        u = paddle.to_tensor(u, dtype="int64")
        seq = paddle.to_tensor(seq, dtype="int64")
        pos = paddle.to_tensor(pos, dtype="int64")
        neg = paddle.to_tensor(neg, dtype="int64")
        time_seq = paddle.to_tensor(time_seq, dtype="int64")
        time_matrix = paddle.to_tensor(time_matrix, dtype="int64")

        pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
        pos_labels, neg_labels = paddle.ones(pos_logits.shape), paddle.zeros(neg_logits.shape)

        adam_optimizer.clear_grad()

        targets = (pos != 0).astype(dtype='int64')
        loss = bce_criterion(pos_logits, neg_logits, targets)

        # l2 panalty of embedding
        for param in model.item_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * paddle.norm(param)

        loss.backward()
        adam_optimizer.step()

        if step % args.log_freq == 0:
            print(
                "Epoch {} iteration {}: {}".format(epoch, step, loss.item()))  # expected 0.4~0.6 after init few epochs
            sys.stdout.flush()

    if epoch % args.eval_freq == 0 and epoch > args.eval_freq * 2:  # skip the first few logs
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating...')
        with paddle.no_grad():
            t_valid = evaluate_valid(model, dataset, args)
            print('Epoch:%d, time: %f(s) \nvalid (NDCG@10: %.4f, HR@10: %.4f)' % (epoch, T, t_valid[0], t_valid[1]))
            t_test = evaluate(model, dataset, args)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()
        model.train()

        folder = args.dataset + '_' + args.train_dir
        fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)

        save_checkpoint(model, {'epoch': epoch,
                                'optimizer': adam_optimizer.state_dict()},
                        os.path.join(folder, fname))

f.close()
sampler.close()
print("Done")
