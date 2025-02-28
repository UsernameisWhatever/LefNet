from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
from datetime import datetime
import warnings
import copy
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import pickle
import pandas as pd
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.args = args
        self.outside_optimizer_use_test_loss = 1e10

        with open(f'./epochs_loss_per_dim/non_zero_indices_333_without_date.pkl', 'rb') as f:
            self.non_zero_indices_list_333 = pickle.load(f)
        with open(f'./epochs_loss_per_dim/non_zero_indices_83_without_date.pkl', 'rb') as f:
            self.non_zero_indices_list_83 = pickle.load(f)

        if args.data == 'agriculture':  # TODO 看下是否有相关参数，是否需要保留
            self.para_file_path = "/mnt/disk0/data/paras_data_extract_with_date/"
            self.sci_counting = np.load('./data/reshape_sci_counting.npy')[:, -83:]
            self.sci_counting = self.sci_counting[:, self.non_zero_indices_list_83]

        else:
            self.sci_counting = 1

        self.mse_loss_ = nn.MSELoss(reduction='none')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss_ = criterion(pred*self.sci_counting, true*self.sci_counting)

                total_loss.append(loss_)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print('start load train data')
        train_data, train_loader = self._get_data(flag='train')
        print('start load vali_data')
        vali_data, vali_loader = self._get_data(flag='val')
        print('start load test_data')
        test_data, test_loader = self._get_data(flag='test')
        print('End loading')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            variance_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                torch.autograd.set_detect_anomaly(True)

                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                    outputs1 = outputs * torch.from_numpy(self.sci_counting).to(self.device)
                    batch_y1 = batch_y * torch.from_numpy(self.sci_counting).to(self.device)

                    loss1 = criterion(outputs1, batch_y1)
                    loss2 = self.mse_loss_(outputs1, batch_y1)
                    loss_varience = torch.mean(((outputs - batch_y) - torch.mean(outputs - batch_y)) ** 2)

                    output_file = f'./epochs_loss_per_dim/{self.args.model}_losses.txt'
                    with open(output_file, 'w') as f:
                        f.write(f"Epoch {epoch + 1}\n")
                        for i__ in range(loss2.shape[2]):
                            f.write(f"Dimension {i__}: Loss = {loss2[:, :, i__].mean().item()}\n")
                        f.write("\n")
                    train_loss.append(loss1.item())
                    variance_loss.append(loss_varience.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            variance_loss = np.average(variance_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.outside_optimizer_use_test_loss = test_loss

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Variance of loss: ", variance_loss.item())
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch + 1 in self.args.prune_epochs:
                torch.save(self.model, f'./temp_models/{epoch}_model_0.pth')
                torch.save(self.model.state_dict(), f'./temp_models/{epoch}_weights_0.pth')
                self.model.prune_().check_pruning_effectiveness()
                print("Model pruned at Epoch: ", epoch + 1)
                torch.save(self.model, f'./temp_models/{epoch}_model_1.pth')
                torch.save(self.model.state_dict(), f'./temp_models/{epoch}_weights_1.pth')

            if (epoch + 1) % 10 == 0:
                self.model.state_report()
                self.model.state_report_dict['first_layer']['epoch'].append(epoch + 1)
                self.model.state_report_dict['last_layer']['epoch'].append(epoch + 1)
                self.model.state_report_dict['first_layer']['loss'].append(round(float(test_loss), 3))
                self.model.state_report_dict['last_layer']['loss'].append(round(float(test_loss), 3))

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return train_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, dtw, rmse, mape, mspe))
        __f = f'{self.args.model}_{datetime.now().strftime("%Y-%m-%d_%H%M")}'
        with open(f'./epochs_loss_per_dim/{__f}.txt', 'w') as file:
            file.write(f"mse:\t{mse}\nmae:\t{mae}\nrmse:\t{rmse}\nmape:\t{mape}\nmspe:\t{mspe}\n")
        print('Real data test results: ')
        preds_ = preds.copy()*self.sci_counting
        trues_ = trues.copy()*self.sci_counting
        mae, mse, rmse, mape, mspe = metric(preds_, trues_)
        print('mse:{}, mae:{}, dtw:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, dtw, rmse, mape, mspe))
        with open(f'./epochs_loss_per_dim/{__f}.txt', 'a') as file:
            file.write(f"\n\nmse:\t{mse}\nmae:\t{mae}\nrmse:\t{rmse}\nmape:\t{mape}\nmspe:\t{mspe}\n")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
