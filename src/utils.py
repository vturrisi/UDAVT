import json
import os
from collections import defaultdict
from os import makedirs, system
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import LogSoftmax

from .gather_layer import gather

try:
    from pytorch_lightning.metrics.functional.classification import confusion_matrix
except:
    from pytorch_lightning.metrics.functional import confusion_matrix

import wandb


# computes entropy of input x
def compute_entropy(x):
    epsilon = 1e-5
    H = -x * torch.log(x + epsilon)
    H = torch.sum(H, dim=1)
    return H


# checkpointer callback to call at given frequency
class EpochCheckpointer(Callback):
    def __init__(self, args, logdir="trained_models", frequency=25):
        self.args = args
        self.frequency = frequency
        self.logdir = logdir

    def initial_setup(self, trainer):
        if self.args.wandb:
            version = str(trainer.logger.version)
        else:
            version = None
        if version is not None:
            self.path = os.path.join(self.logdir, version)
            self.ckpt_placeholder = f"{self.args.name}-{version}" + "-ep={}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.args.name}" + "-ep={}.ckpt"

        # create logging dirs
        if trainer.is_global_zero:
            try:
                os.makedirs(self.path)
            except:
                pass

    def save_args(self, trainer):
        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = os.path.join(self.path, "args.json")
            json.dump(args, open(json_path, "w"))

    def save(self, trainer):
        epoch = trainer.current_epoch
        ckpt = self.ckpt_placeholder.format(epoch)
        trainer.save_checkpoint(os.path.join(self.path, ckpt))

    def on_train_start(self, trainer, pl_module):
        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.frequency == 0 and epoch != 0:
            self.save(trainer)

    def on_train_end(self, trainer, pl_module):
        self.save(trainer)


class TSNECallback(Callback):
    def __init__(self, wandb=False):
        self.wandb = wandb

    def on_train_epoch_end(self, trainer, module, outputs):
        batches_data = outputs[0]

        data_source = []
        Y_source = []

        data_target = []
        Y_target = []

        for batch_data in batches_data:
            batch_data = batch_data[0]["extra"]

            feat_s = batch_data["feat_s"]
            feat_s = gather(feat_s).detach().cpu()
            y_source = batch_data["y_source"]
            y_source = gather(y_source).detach().cpu()

            feat_t = batch_data["feat_t"]
            feat_t = gather(feat_t).detach().cpu()
            y_target = batch_data["y_target"]
            y_target = gather(y_target).detach().cpu()

            data_source.append(feat_s)
            Y_source.append(y_source)

            data_target.append(feat_t)
            Y_target.append(y_target)

        if trainer.is_global_zero and len(data_target):
            data_source = torch.cat(data_source, dim=0).numpy()
            data_target = torch.cat(data_target, dim=0).numpy()
            Y_source = torch.cat(Y_source, dim=0).numpy()
            Y_target = torch.cat(Y_target, dim=0).numpy()
            Y = torch.cat((Y_source, Y_target))

            data = TSNE(n_components=2).fit_transform(torch.cat((data_source, data_target)))
            # assing to dataframe
            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["y"] = Y
            df["domains"] = ["source"] * data_source.size(0) + ["target"] * data_target.size(0)

            plt.figure(figsize=(16, 10))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="y",
                style="domains",
                palette=sns.color_palette("hls", len(np.unique(Y))),
                data=df,
                legend="full",
                alpha=0.3,
            )
            plt.tight_layout()
            if self.args.wandb:
                wandb.log(
                    {f"target_tsne": wandb.Image(ax)},
                    commit=False,
                )
            else:
                plt.savefig("tsne.jpg")
            plt.close()

# handles generation of confusion matrix
class ConfusionMatrix(Callback):
    def __init__(self, args):
        self.args = args

    def on_train_epoch_start(self, trainer, module):
        self.outputs_s = []
        self.targets_s = []

        self.outputs_t = []
        self.targets_t = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for gpu_outputs in outputs:
            for output in gpu_outputs:
                output = output["extra"]
                if "out_s" in output:
                    self.outputs_s.append(output["out_s"].cpu())
                    self.targets_s.append(output["y_source"].cpu())

                if "out_t" in output:
                    self.outputs_t.append(output["out_t"].cpu())
                    self.targets_t.append(output["y_target"].cpu())

    def on_train_epoch_end(self, trainer, module, outputs):
        if trainer.is_global_zero:
            for name, outputs, targets in zip(
                ["source", "target"],
                [self.outputs_s, self.outputs_t],
                [self.targets_s, self.targets_t],
            ):
                if len(outputs):
                    outputs = torch.cat(outputs)
                    targets = torch.cat(targets)

                    preds = outputs.float().max(dim=1)[1]

                    cm = confusion_matrix(preds, targets, num_classes=module.num_classes).cpu()
                    sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                    sns.set(font_scale=2)
                    ax = sns.heatmap(data=cm, annot=True, cmap="OrRd")
                    values = list(range(cm.size(0)))
                    ax.set_xticklabels(values, rotation=45, fontsize="large")
                    ax.set_yticklabels(values, rotation=90, fontsize="large")
                    plt.tight_layout()
                    if self.args.wandb:
                        wandb.log({f"train_{name}_cm": wandb.Image(ax)}, commit=False)
                        plt.close()

    def on_validation_epoch_start(self, trainer, module):
        self.outputs = []
        self.targets = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.outputs.append(outputs["outputs"].cpu())
        self.targets.append(outputs["targets"].cpu())

    def on_validation_epoch_end(self, trainer, module):
        if trainer.is_global_zero:
            self.outputs = torch.cat(self.outputs)
            self.targets = torch.cat(self.targets)

            preds = self.outputs.float().max(dim=1)[1]
            targets = self.targets

            cm = confusion_matrix(preds, targets, num_classes=module.num_classes).cpu()
            if cm.size():
                sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                sns.set(font_scale=2)
                ax = sns.heatmap(data=cm, annot=True, cmap="OrRd")
                values = list(range(cm.size(0)))
                ax.set_xticklabels(values, rotation=45, fontsize="large")
                ax.set_yticklabels(values, rotation=90, fontsize="large")
                plt.tight_layout()
                if self.args.wandb:
                    wandb.log({"val_cm": wandb.Image(ax)}, commit=False)
                    plt.close()


# computes the accuracy over the k top predictions for the specified values of k
def accuracy_at_k(output, target, top_k=(1, 5)):
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_acc(y, y_hat, device):
    acc = y == y_hat
    if len(acc) > 0:
        acc = acc.sum().detach().true_divide(acc.size(0))
    else:
        acc = torch.tensor(0.0, device=device)

    return acc


def soft_ce(logits, target):
    logs = LogSoftmax(dim=1)(logits)
    loss = -torch.sum(logs * target, dim=1).mean()
    return loss


def plot_pseudo_labels(out_clips, n_labels, gt):

    plt.close()

    labels = [str(l) for l in range(n_labels)]

    if exists("plots"):
        system("rm -r plots")
    makedirs("plots")
    makedirs("plots/hard/")
    makedirs("plots/soft/")

    with torch.no_grad():
        softmax = F.softmax(out_clips, dim=0).sum(dim=1)

    i = 0
    for s in softmax:
        x = range(n_labels)
        y = [j for j in s]
        x_pos = [j for j, _ in enumerate(x)]
        plt.plot(y, color="green")
        plt.xlabel("Labels")
        plt.ylabel("Density")
        plt.title("Video {} - gt = {}".format(i, gt[i]))
        plt.xticks(x_pos, x)
        plt.savefig("plots/soft/{}.png".format(i))
        plt.close()
        i += 1

    predictions = out_clips.argmax(dim=2)
    occurrences = {}

    i = 0
    for p in predictions:
        occurrences[i] = {}
        for l in labels:
            occurrences[i][l] = 0
        i += 1

    i = 0
    for p in predictions:
        for l in p:
            occurrences[i][str(l.item())] += 1
        i += 1
    y = [v for _, v in occurrences.items()]
    for i, v in enumerate(y):
        x = v.keys()
        y = v.values()
        x_pos = [j for j, _ in enumerate(x)]
        plt.bar(x_pos, y, color="green")
        plt.xlabel("Labels")
        plt.ylabel("Occurrences")
        plt.title("Video {} - gt = {}".format(i, gt[i]))
        plt.xticks(x_pos, x)
        plt.savefig("plots/hard/{}.png".format(i))
        plt.close()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low
    )


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def weighted_mean(outputs, key, batch_size_key):
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd2(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError("ver == 1 or 2")

    return loss
