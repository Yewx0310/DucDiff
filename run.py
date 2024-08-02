# -*- coding: utf-8 -*-
import datetime
import numpy as np
import Constants
import torch
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader
from dataLoader import datasets, Read_data, Split_data
from parsers import parser
from utils import EarlyStopping
from Metrics import Metrics
from model import DiffCas

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn._reduction")

metric = Metrics()
opt = parser.parse_args()

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def init_seeds(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def get_correct(pred, gold):
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return n_correct


def model_training(model, train_loader, epoch):
    ''' model training '''

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    print('start training: ', datetime.datetime.now())
    model.train()

    for step, (cascade_item, label, cascades_fru, cascade_time, label_time, cascade_len) in enumerate(train_loader):
        n_words = label.data.ne(Constants.PAD).sum().float().item()
        n_total_words += n_words
        model.zero_grad()
        cascade_item = trans_to_cuda(cascade_item.long())
        tar = trans_to_cuda(label.long())

        past_output_noib_mask, past_output_noib, future_output_noib, past_output_ib, future_output_ib = model(cascade_item, tar)
        
        tar = tar.contiguous().view(-1)
        cascade_item = cascade_item.contiguous().view(-1)

        ce_loss = 0.7 * model.loss_ce(past_output_noib_mask, tar) + \
                  0.3 * model.loss_ce(future_output_noib, cascade_item)

        vcd_loss = 0.5 * kl_div(input=model.softmax(past_output_noib.detach()),
                                target=model.softmax(future_output_ib),reduction="sum") + \
                0.5 * kl_div(input=model.softmax(future_output_noib.detach()),
                                target=model.softmax(past_output_ib),reduction="sum")

        loss = ce_loss + 0.3 * vcd_loss
        n_correct = get_correct(past_output_noib, tar)

        loss.backward()
        model.optimizer.step()
        model.optimizer.update_learning_rate()
        total_loss += loss.item()
        n_total_correct += n_correct

    print('\tTotal Loss:\t%.3f' % total_loss)

    return total_loss, n_total_correct / n_total_words


def model_testing(model, test_loader, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0.0
    n_correct = 0.0

    print('start predicting: ', datetime.datetime.now())
    model.eval()

    with torch.no_grad():
        for step, (cascade_item, label, cascades_fru, cascade_time, label_time, cascade_len) in enumerate(test_loader):

            cascade_item = trans_to_cuda(cascade_item.long())
            y_pred = model.model_prediction(cascade_item)

            y_pred = y_pred.detach().cpu()
            tar = label.view(-1).detach().cpu()

            pred = y_pred.max(1)[1]
            gold = tar.contiguous().view(-1)
            correct = pred.data.eq(gold.data)
            n_correct = correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            scores_batch, scores_len = metric.compute_metric(y_pred, tar, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores, n_correct / n_total_words


def train_test(epoch, model, train_loader, val_loader, test_loader):
    total_loss, accuracy = model_training(model, train_loader, epoch)
    val_scores, val_accuracy = model_testing(model, val_loader)
    test_scores, test_accuracy = model_testing(model, test_loader)

    return total_loss, val_scores, test_scores, val_accuracy.item(), test_accuracy.item()


def main(data_path):
    init_seeds(opt.seed)

    if opt.preprocess:
        Split_data(data_path, train_rate=opt.train_rate, valid_rate=opt.valid_rate, load_dict=True)

    train, valid, test, user_size = Read_data(data_path)

    train_data = datasets(train, opt.max_len)
    val_data = datasets(valid, opt.max_len)
    test_data = datasets(test, opt.max_len)

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    opt.user_size = user_size

    save_model_path = opt.save_path
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=save_model_path)

    model = trans_to_cuda(DiffCas(opt))

    top_K = [10, 50, 100]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    validation_history = 0.0

    for epoch in range(opt.epochs):
        total_loss, val_scores, test_scores, val_accuracy, test_accuracy = train_test(epoch, model, train_loader,
                                                                                      val_loader, test_loader)

        if validation_history <= sum(val_scores.values()):
            validation_history = sum(val_scores.values())

            for K in top_K:
                test_scores['hits@' + str(K)] = test_scores['hits@' + str(K)] * 100
                test_scores['map@' + str(K)] = test_scores['map@' + str(K)] * 100

                best_results['metric%d' % K][0] = test_scores['hits@' + str(K)]
                best_results['epoch%d' % K][0] = epoch
                best_results['metric%d' % K][1] = test_scores['map@' + str(K)]
                best_results['epoch%d' % K][1] = epoch

            print(" -validation scores:-------------------------------------")
            print('  - (validation) accuracy: {accu:3.3f} %'.format(accu=100 * val_accuracy))
            for metric in val_scores.keys():
                print(metric + ' ' + str(val_scores[metric] * 100))

            print(" -test scores:-------------------------------------")
            print('  - (testing) accuracy: {accu:3.3f} %'.format(accu=100 * test_accuracy))
            for K in top_K:
                print('train_loss:\t%.4f\tRecall@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
                      (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                       best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
        early_stopping(-sum(list(val_scores.values())), model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    print(" -(Finished!!) \n parameter settings: ")
    print("--------------------------------------------")
    print(opt)

    print(" -(Finished!!) \n test scores: ")
    print("--------------------------------------------")
    for K in top_K:
        print('Recall@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
              (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
               best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == "__main__":
    opt.save_path = f"./checkpoint/{opt.data_name}/lr_{opt.lr}_batch_{opt.batch_size}_step_{opt.steps}_emb_{opt.hidden_size}.pt"
    opt.compress_emb = int(opt.hidden_size * 0.75)
    print(opt)
    main(opt.data_name)
