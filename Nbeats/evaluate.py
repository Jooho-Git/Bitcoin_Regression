import torch
import time
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
import os
from scipy.stats import norm
from scipy.stats import t

#-- Visualization
import plotly.express as px
import plotly.io as pio
import kaleido
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ModelTest:
    '''
    Model Testing with MC Dropout (VB-consistent predictive intervals)
    '''

    def __init__(
        self,
        model,
        test_dataloader,
        yscaler,
        metric,
        n_samples,
        input_window,
        output_window,
        log_dir
    ):
        self.model = model
        self.testloader = test_dataloader
        self.yscaler = yscaler
        self.metric = metric
        self.n_samples = n_samples
        self.input_window = input_window
        self.output_window = output_window
        self.log_dir = log_dir

        self.tau_inv = 0.0  # Default: 0 (no aleatoric variance)

        test_loss, true_list, y_pred_list, trend_list, seasonal_list = self.test()

        self.history = {}
        self.history['test'] = test_loss

        true, y_pred, trend, seasonality = self.reshape_data(true_list, y_pred_list, trend_list, seasonal_list)
        self.plot_prediction(true, y_pred, trend, seasonality)

    def set_tau_from_training(self, keep_prob: float, weight_decay: float, n_train: int, lengthscale: float):
        """
        tau = (l^2 * p) / (2 * N * lambda), where p=keep_prob=1-drop_prob
        -> tau_inv = 1/tau
        """
        tau = (lengthscale ** 2 * keep_prob) / (2.0 * n_train * weight_decay)
        self.tau_inv = float(1.0 / max(tau, 1e-12))

    def test(self):
        """
        Monte Carlo Dropout Test:
        - Keep BatchNorm fixed (eval mode)
        - Enable dropout layers for stochastic inference
        """
        self.model.eval()

        DROPOUT_TYPES = (
            torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
            torch.nn.Dropout3d, torch.nn.AlphaDropout
        )
        def enable_dropout(m):
            if isinstance(m, DROPOUT_TYPES):
                m.train()

        def inv_tr(scaler, arr):
            """Fix shape for inverse_transform: (...,) -> (-1,1) -> restored shape"""
            arr = np.asarray(arr)
            flat = arr.reshape(-1, 1)
            out = scaler.inverse_transform(flat)
            return out.reshape(arr.shape)

        loss_lst = []
        true_list = []
        y_pred_list = []
        trend_list = []
        seasonal_list = []

        for i, (x_test, y_test) in enumerate(self.testloader):
            preds_mc = []
            trend_mc = []
            seasonal_mc = []

            for n in range(self.n_samples):
                self.model.apply(enable_dropout)
                with torch.no_grad():
                    backcast, y_pred, trend_forecast, seasonal_forecast = self.model(
                        x_test.to(self.model.device)
                    )

                preds_mc.append(y_pred.squeeze(0).detach().cpu().numpy())
                trend_mc.append(trend_forecast.squeeze(0).detach().cpu().numpy())
                seasonal_mc.append(seasonal_forecast.squeeze(0).detach().cpu().numpy())

            preds_mc = np.stack(preds_mc, axis=0)
            trend_mc = np.stack(trend_mc, axis=0)
            seasonal_mc = np.stack(seasonal_mc, axis=0)

            y_test = y_test.squeeze(0).detach().cpu().numpy()
            x_test = x_test.squeeze(0).detach().cpu().numpy()

            preds_mc = inv_tr(self.yscaler, preds_mc)
            y_test = inv_tr(self.yscaler, y_test).ravel()
            x_test = inv_tr(self.yscaler, x_test).ravel()
            trend_mc = inv_tr(self.yscaler, trend_mc)
            seasonal_mc = inv_tr(self.yscaler, seasonal_mc)

            true = np.concatenate([x_test, y_test])

            pred_mean = preds_mc.mean(axis=0).ravel()
            y_vec = y_test.ravel()
            x_vec = x_test.ravel()

            if self.metric == 'MAPE':
                loss = MAPEval(pred_mean, y_vec)
            elif self.metric == 'MASE':
                loss = MASEval(x_vec, y_vec, pred_mean)
            elif self.metric == 'ALL':
                loss = (
                    MAPEval(pred_mean, y_vec),
                    SMAPEval(pred_mean, y_vec),
                    MAEval(pred_mean, y_vec),
                )
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            loss_lst.append(loss)

            if i % 20 == 0:
                true_list.append(true)
                y_pred_list.append(preds_mc)
                trend_list.append(trend_mc)
                seasonal_list.append(seasonal_mc)

        loss_arr = np.array(loss_lst, dtype=object)
        if self.metric == 'ALL':
            mape = np.mean([x[0] for x in loss_lst])
            smape = np.mean([x[1] for x in loss_lst])
            mae = np.mean([x[2] for x in loss_lst])
            test_loss = (mape, smape, mae)
        else:
            test_loss = np.mean([float(x) for x in loss_lst])

        print(f"Average test {self.metric}: {test_loss}")

        return test_loss, true_list, y_pred_list, trend_list, seasonal_list

    def plot_prediction(self, true, y_pred, trend, seasonality):
        save_directory = f'{self.log_dir}/predict_img'
        os.makedirs(save_directory, exist_ok=True)

        for j in tqdm(range(len(true))):
            true_seq = np.asarray(true[j]).ravel()
            preds = np.asarray(y_pred[j])
            trds  = np.asarray(trend[j])
            seas  = np.asarray(seasonality[j])

            H = preds.shape[1]
            L = len(true_seq) - H
            assert H > 0 and L >= 0

            pred_mean, pred_lower, pred_upper = self.confidence_interval(preds, alpha=0.01, tau_inv=self.tau_inv)
            trend_mean, trend_lower, trend_upper = self.confidence_interval(trds,  alpha=0.01, tau_inv=self.tau_inv)
            seasonal_mean, seasonal_lower, seasonal_upper = self.confidence_interval(seas, alpha=0.01, tau_inv=self.tau_inv)

            fig = make_subplots(
                subplot_titles=['True Vs Predicted', 'Trend', 'Seasonality'],
                rows=2, cols=2,
                vertical_spacing=0.12,
                horizontal_spacing=0.07,
                column_widths=[0.75, 0.25],
                row_heights=[0.6, 0.4],
                specs=[[{"rowspan": 2}, {}], [None, {}]]
            )

            t_hist = list(range(L))
            t_fore = list(range(L, L + H))

            fig.add_trace(go.Scatter(
                x=list(range(L + H)),
                y=true_seq.tolist(),
                name="True",
                mode="lines"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t_fore,
                y=pred_mean.tolist(),
                name="Prediction mean",
                mode="lines"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t_fore + t_fore[::-1],
                y=pred_upper.tolist() + pred_lower[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(179,226,205,0.35)',
                line=dict(width=0),
                name='99% PI',
                showlegend=True
            ), row=1, col=1)

            fig.add_vline(x=L-1, line_width=1, line_dash="dash", opacity=0.5, row=1, col=1)

            fig.add_trace(go.Scatter(
                x=t_fore,
                y=trend_mean.tolist(),
                name="Trend mean",
                mode="lines"
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=t_fore + t_fore[::-1],
                y=trend_upper.tolist() + trend_lower[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(66, 165, 245, 0.25)',
                line=dict(width=0),
                name='Trend 99% PI',
                showlegend=False
            ), row=1, col=2)

            fig.add_trace(go.Scatter(
                x=t_fore,
                y=seasonal_mean.tolist(),
                name="Seasonal mean",
                mode="lines"
            ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=t_fore + t_fore[::-1],
                y=seasonal_upper.tolist() + seasonal_lower[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(156, 204, 101, 0.25)',
                line=dict(width=0),
                name='Seasonal 99% PI',
                showlegend=False
            ), row=2, col=2)

            fig.update_layout(
                height=550, width=1200,
                title_text=f"Prediction (MC Dropout) — sample {j}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

            pio.write_image(fig, f'{save_directory}/nbeats_pred_{j}.png', format='png', engine='kaleido')

        if len(true) > 0:
            fig.show()

    def reshape_data(self, true_list, y_pred_list, trend_list, seasonal_list):
        true = [np.asarray(t).ravel() for t in true_list]
        y_pred = [np.asarray(p) for p in y_pred_list]
        trend = [np.asarray(tr) for tr in trend_list]
        seasonality = [np.asarray(se) for se in seasonal_list]
        return true, y_pred, trend, seasonality

    def confidence_interval(self, mc_samples, alpha=0.01, tau_inv=0.0):
        """
        Compute the prediction interval:
            [mean - z * sqrt(s^2 + 1/tau),  mean + z * sqrt(s^2 + 1/tau)]
        Returns:
            mean : np.ndarray (H,)
            lower : np.ndarray (H,)
            upper : np.ndarray (H,)
        """
        mc_samples = np.asarray(mc_samples)
        assert mc_samples.ndim == 2, "mc_samples must be (T, H)"

        mean = mc_samples.mean(axis=0)
        s2 = ((mc_samples - mean) ** 2).mean(axis=0)
        sigma_hat = np.sqrt(s2 + tau_inv)
        z = norm.ppf(1 - alpha / 2)

        lower = mean - z * sigma_hat
        upper = mean + z * sigma_hat
        return mean, lower, upper
