import os
import pickle as pkl
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from model import CaRaCountModel
from utils import *


PHASE_COLOR = {
    0: "gray",
    1: "lightsteelblue",
    2: "salmon",
    3: "peachpuff",
    4: "seashell",
    5: "gray",
}


class CaRaCount():
    def __init__(
            self,
            num_phase: int, # exclude background class
            examplar,
            examplar_cnt,
            query,
            device: str,
        ):
        
        self.num_phase = num_phase
        self.device = device

        self.examplar = examplar
        self.examplar_cnt = examplar_cnt
        self.query = query

        # preprocess examplar
        examplar_processed, self.plot_examplar, self.scaler = preprocessing(examplar)
        est_rep_duration = get_rep_duration(examplar_processed, num_phase)
        self.window_size = int(est_rep_duration/num_phase)
        self.stride = int(self.window_size/2)

        examplar_wd = sliding_windows(examplar_processed, self.window_size, self.stride)
        examplar_wd = torch.tensor(examplar_wd, dtype=torch.float32)
        self.examplar_wd = (
            torch.tensor(examplar_wd, dtype=torch.float32)
                .unsqueeze(0)            # (1, num_win, window_size, 6)
                .permute(0, 3, 1, 2)     # (1, 6, num_win, window_size)
        )

        # preprocess examplar
        query_processed, self.plot_query, _ = preprocessing(query, self.scaler)
        query_wd = sliding_windows(query_processed, self.window_size, self.stride)
        query_wd = torch.tensor(query_wd, dtype=torch.float32)
        self.query_wd = (
            torch.tensor(query_wd, dtype=torch.float32)
                .unsqueeze(0)            # (1, num_win, window_size, 6)
                .permute(0, 3, 1, 2)     # (1, 6, num_win, window_size)
        )

        # construct target sequence
        num_background = get_background_window(est_rep_duration, num_phase, examplar_processed.shape[0], self.window_size)
        background_class = num_phase + 1
        target_seq = [phase for _ in range(examplar_cnt) for phase in range(1, num_phase + 1)]
        target_seq = [background_class] * num_background + target_seq + [background_class] * num_background
        self.target_seq = torch.tensor(target_seq, dtype=torch.long).view(-1)

        # construct model and optimizer
        set_seed(1234)
        self.model = CaRaCountModel(num_phase+1, self.window_size)
        self.lr = 8e-3
        self.weight_decay = 3e-8
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CTCLoss()

    def fit(self, 
            torelance=2000,
            verbose=None,
            save_dir=None):
        
        self.examplar_wd = self.examplar_wd.to(self.device)
        self.target_seq = self.target_seq.to(self.device)
        self.model = self.model.to(self.device)
        self.model.train()

        input_len = torch.full((self.examplar_wd.shape[0],), self.examplar_wd.shape[2], dtype=torch.long)
        target_len = torch.as_tensor(len(self.target_seq)).view(-1)

        ctc_loss = 1000
        cur_iter = 0
        while ctc_loss > 0.05:
            output = self.model(self.examplar_wd.float())
            ctc_input = F.log_softmax(output, dim=2)
            ctc_input = ctc_input.transpose(0, 1)
            loss = self.criterion(ctc_input,  self.target_seq, input_len, target_len)
            ctc_loss = loss.item()

            self.optimizer.zero_grad()   
            loss.backward()
            self.optimizer.step()

            preds = F.softmax(output,dim=2) 
            preds = preds.squeeze(0)
            phase_cls = torch.max(preds, dim = len(preds.shape)-1)
            phase_cls = phase_cls.indices
            self.examplar_preds = [p.item() for p in phase_cls]

            if verbose is not None and (cur_iter+1)%verbose==0:
                print('Iter {}/{}: loss = {:.4f}'.format(cur_iter,torelance,loss.item()))
            if cur_iter > torelance:
                break
            cur_iter += 1

        print(f"Done fitting examplar at iter: {cur_iter} | ctc loss {ctc_loss}")
        if save_dir is not None:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
            pkl.dump(self.ex_scaler, open(os.path.join(save_dir, "scaler.sav"), 'wb'))
            dict ={
                "num_phase" : self.num_phase,
                "window_size" : self.window_size,
                "stride" : self.stride,
            }
            with open(os.path.join(save_dir, "data_dict.json"), "w") as f:
                json.dump(dict, f)
            print(f"Saving to {save_dir}")


    def predict(self):
        self.query_wd = self.query_wd.to(self.device)
        self.model.train()

        with torch.no_grad():
            output = self.model(self.query_wd.float())
            output = torch.squeeze(output)
            output = output.unsqueeze(0)
            preds = F.softmax(output, dim=2)
            preds = preds.squeeze(0)
            phase_cls = torch.max(preds[:,:], dim = len(preds.shape)-1)
            phase_cls = phase_cls.indices

            self.query_preds = [p.item() for p in phase_cls]
            query_preds_format = ctc_decode(self.query_preds)
            final_count = get_count(query_preds_format, self.num_phase)
        return final_count


    def visualize(self, examplar=True):
            plot_data = self.plot_examplar if examplar else self.plot_query
            signal_len = len(plot_data)
            phases_list = self.examplar_preds if examplar else self.query_preds

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(plot_data)

            for i in range(len(phases_list)):
                start = i * self.stride
                end = min(start + self.window_size, signal_len)
                lbl = phases_list[i]
                ax.axvspan(start, end, color=PHASE_COLOR[lbl], alpha=0.5)

                if examplar:
                    if lbl == 4 and phases_list[i+1] == 1:
                        boundary = (i + 1) * self.stride
                        ax.axvline(x=boundary, color="r", linestyle="--", linewidth=2)

            unique_labels = np.unique(phases_list)
            legend_patches = []
            for lbl in unique_labels:
                label = f"Phase {lbl}" if lbl in [1, 2, 3, 4] else "Background" 
                patch = mpatches.Patch(color=PHASE_COLOR[lbl], label=label)
                legend_patches.append(patch)

            handles, texts = plt.gca().get_legend_handles_labels()
            handles.extend(legend_patches)
            text_list = [ "Background", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
            texts.extend(text_list)

            ax.set_xticks([])
            ax.set_yticks([]) 
            for spine in ax.spines.values():
                spine.set_visible(True)
            plt.legend(handles, texts, loc="upper right")
            title = "Fitting results on Examplar sequence" if examplar else \
                "Predicting results on Query sequence"
            plt.title(title)
            plt.show()