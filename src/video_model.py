import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import wandb
from einops import rearrange

try:
    from adversarial import DomainClassifier
    from gather_layer import gather
    from info_bottleneck import IBHead, compute_ib_loss
    from modules import CDAN, AdversarialNetworkCDAN, RandomLayer
    from simclr import SimCLRHead, compute_simclr_loss
    from utils import accuracy_at_k, compute_acc, mmd2
    from vicreg import VICRegHead, compute_vicreg_loss
except ImportError:
    from .adversarial import DomainClassifier
    from .gather_layer import gather
    from .info_bottleneck import IBHead, compute_ib_loss
    from .modules import CDAN, AdversarialNetworkCDAN, RandomLayer
    from .simclr import SimCLRHead, compute_simclr_loss
    from .utils import accuracy_at_k, compute_acc, mmd2
    from .vicreg import VICRegHead, compute_vicreg_loss


def static_lr(get_lr, param_group_indexes, lrs_to_replace):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


def weighted_mean(outputs, key, batch_size_key):
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


class ClassMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers):
        super().__init__()

        if n_layers:
            self.model = []
            self.model.append(nn.Linear(in_dim, hidden_dim))
            self.model.append(nn.BatchNorm1d(hidden_dim))
            self.model.append(nn.ReLU(hidden_dim))
            for _ in range(n_layers - 1):
                self.model.append(nn.Linear(hidden_dim, hidden_dim))
                self.model.append(nn.BatchNorm1d(hidden_dim))
                self.model.append(nn.ReLU(hidden_dim))
            self.model = nn.Sequential(*self.model)
            self.classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.model = nn.Identity()
            self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        feat = self.model(x)
        out = self.classifier(feat)
        return feat, out


class TransformerVideoModel(pl.LightningModule):
    def __init__(
        self,
        transformer,
        num_classes,
        args,
    ):
        super().__init__()

        # define base model
        self.transformer = transformer
        self.num_classes = num_classes
        self.args = args

        if args.use_queue:
            # queue
            self.queue_size = args.queue_size
            self.register_buffer("source_queue", torch.randn(self.queue_size, args.mlp_hidden_dim))
            self.register_buffer("target_queue", torch.randn(self.queue_size, args.mlp_hidden_dim))
            self.register_buffer("source_queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("target_queue_y", -torch.ones(self.queue_size, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # replace head with mlp
        if args.replace_with_mlp:
            self.transformer.head = ClassMLP(
                self.transformer.embed_dim, args.mlp_hidden_dim, num_classes, args.mlp_n_layers
            )

        # transfer loss
        if args.da == "cdan":
            self.CDAN_domain_classifier = AdversarialNetworkCDAN(
                in_feature=args.mlp_hidden_dim,
                hidden_size=args.mlp_hidden_dim,
            )
            self.random_layer = RandomLayer(
                [args.mlp_hidden_dim, self.num_classes], args.mlp_hidden_dim
            )

        elif args.da == "adversarial":
            if args.mlp_n_layers:
                in_dim = args.mlp_hidden_dim
            else:
                in_dim = self.transformer.embed_dim
            self.domain_classifier = DomainClassifier(in_dim, args.mlp_hidden_dim)

        elif args.da == "ib":
            if args.mlp_n_layers:
                in_dim = args.mlp_hidden_dim
            else:
                in_dim = self.transformer.embed_dim

            self.contrastive_head = IBHead(in_dim, args.mlp_hidden_dim, args.mlp_hidden_dim)

        elif args.da == "vicreg":
            if args.mlp_n_layers:
                in_dim = args.mlp_hidden_dim
            else:
                in_dim = self.transformer.embed_dim

            self.contrastive_head = VICRegHead(in_dim, args.mlp_hidden_dim, 256)

        elif args.da == "simclr":
            if args.mlp_n_layers:
                in_dim = args.mlp_hidden_dim
            else:
                in_dim = self.transformer.embed_dim

            self.contrastive_head = SimCLRHead(in_dim, args.mlp_hidden_dim, 256)

        self.set_training()

    @torch.no_grad()
    def dequeue_and_enqueue(self, z_s, z_t, y_s, y_t):
        z_s = gather(z_s)
        y_s = gather(y_s)
        z_t = gather(z_t)
        y_t = gather(y_t)

        batch_size = z_s.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.source_queue[ptr : ptr + batch_size, :] = z_s
        self.source_queue_y[ptr : ptr + batch_size] = y_s
        self.target_queue[ptr : ptr + batch_size, :] = z_t
        self.target_queue_y[ptr : ptr + batch_size] = y_t
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def configure_optimizers(self):
        args = self.args

        # select optimizer
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD
            extra_optimizer_args = {"momentum": 0.9}
        else:
            optimizer = torch.optim.Adam
            extra_optimizer_args = {}

        # filter parameters to train
        if args.train == "head":
            parameters = self.transformer.head.parameters()

        elif args.train == "head+partial":
            to_keep = ["norm", "head", "pos_embed", "cls_token", "patch_embed"]
            parameters = []
            for name, p in self.named_parameters():
                if any(keep_name in name for keep_name in to_keep):
                    parameters.append(p)

        elif args.train == "head+temporal":
            parameters = list(self.transformer.head.parameters()) + list(
                self.transformer.aggregate.parameters()
            )

        elif args.train == "head+temporal-partial":
            to_keep = ["norm", "pos_embed", "cls_token", "patch_embed"]
            parameters = list(self.transformer.head.parameters())
            for name, p in self.transformer.aggregate.named_parameters():
                if any(keep_name in name for keep_name in to_keep):
                    parameters.append(p)

        elif args.train == "all":
            parameters = self.transformer.parameters()

        else:
            raise ValueError(f"{args.train} not in (head, head+partial, everything)")

        if args.da == "cdan":
            parameters += list(self.CDAN_domain_classifier.parameters())

        elif args.da == "adversarial":
            parameters = [{"params": parameters}]
            parameters += [{"params": self.domain_classifier.parameters(), "lr": args.lr / 2}]

        optimizer = optimizer(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            **extra_optimizer_args,
        )

        # select scheduler
        if args.scheduler == "none":
            return optimizer
        else:
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            elif args.scheduler == "reduce":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            elif args.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps)
            elif args.scheduler == "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.weight_decay)

            return [optimizer], [scheduler]

    def set_training(self):
        if self.args.train == "head":
            self.transformer.eval()
            self.transformer.head.train()
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.transformer.head.parameters():
                param.requires_grad = True

        elif self.args.train == "head+partial":
            to_keep = ["norm", "head", "pos_embed", "cls_token", "patch_embed"]
            for name, p in self.named_parameters():
                if not any(keep_name in name for keep_name in to_keep):
                    p.requires_grad = False

        elif self.args.train == "head+temporal":
            self.transformer.eval()
            self.transformer.head.train()
            self.transformer.aggregate.train()

            for param in self.transformer.parameters():
                param.requires_grad = False

            to_train = list(self.transformer.head.parameters()) + list(
                self.transformer.aggregate.parameters()
            )
            for param in to_train:
                param.requires_grad = True

        elif self.args.train == "head+temporal-partial":
            self.transformer.eval()
            self.transformer.head.train()
            self.transformer.aggregate.train()

            for param in self.transformer.parameters():
                param.requires_grad = False

            for param in self.transformer.head.parameters():
                param.requires_grad = True
            to_keep = ["norm", "pos_embed", "cls_token", "patch_embed"]
            for name, p in self.transformer.aggregate.named_parameters():
                if not any(keep_name in name for keep_name in to_keep):
                    p.requires_grad = False

        if hasattr(self, "contrastive_head"):
            for param in self.contrastive_head.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        self.set_training()

    def forward(self, x):
        x = rearrange(x, "b n_clips c f h w -> (b n_clips f) c h w")
        feat, out = self.transformer(x)
        return feat, out

    def forward_att(self, x):
        b = x.size(0)
        outs = []
        atts = []
        for clip in range(x.size(1)):
            partial = x[:, clip]
            partial = rearrange(partial, "b c f h w -> (b f) c h w")
            out, att = self.transformer.forward_att(partial)
            att = att[:, 0, 1:]
            out = rearrange(out, "(b n_clips) f -> b n_clips f", b=b).detach()
            att = rearrange(att, "(b n_clips) f -> b n_clips f", b=b).detach()

            outs.append(out)
            atts.append(att)

        out = torch.cat(outs, dim=1)
        att = torch.cat(atts, dim=1)
        return out, att

    def single_domain_training_step(self, X, y):
        args = self.args
        log = {}

        # apply model
        _, out = self(X)

        if args.pseudo_labels:
            # get target pseudo-label
            pseudo_y = out.detach().argmax(dim=1)

            if self.args.supervised_labels:
                loss = F.cross_entropy(out, y)
            else:
                loss = F.cross_entropy(out, pseudo_y)

            # compute pseudo-labels accuracies and number of pseudo-labels
            pseudo_labels_acc = compute_acc(y, pseudo_y, self.device)

            # update log
            log["pseudo_labels_acc"] = pseudo_labels_acc

        else:
            loss = F.cross_entropy(out, y)

        acc1, acc5 = accuracy_at_k(out, y, top_k=(1, 5))
        log.update({"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5})

        return loss, log

    def multi_domain_training_step(self, X_source, y_source, X_target, y_target, batch_idx):
        args = self.args

        log = {}

        # apply model
        feat_s, out_s = self(X_source)
        feat_t, out_t = self(X_target)

        # obtain merged features and outputs
        feat = torch.cat((feat_s, feat_t), dim=0)
        out = torch.cat((out_s, out_t), dim=0)
        softmax_out = F.softmax(out, dim=1)

        source_ce_loss = F.cross_entropy(out_s, y_source)
        loss = args.source_ce_loss_weight * source_ce_loss
        log["train_source_ce_loss"] = source_ce_loss

        if not args.source_only:
            # compute pseudolabels if needed
            if args.pseudo_labels:
                with torch.no_grad():
                    pseudo_y = out_t.detach().argmax(dim=1)

                target_ce_loss = F.cross_entropy(out_t, pseudo_y)

                # pseudolabels statistics
                pseudo_labels_acc = compute_acc(y_target, pseudo_y, self.device)

                temp = F.softmax(out_t, dim=1).topk(2, dim=1)[0]
                diff_top2 = (temp[:, 0] - temp[:, 1]).mean()
                log.update(
                    {
                        "pseudo_labels_acc": pseudo_labels_acc,
                        "n_unique_pseudo_labels": pseudo_y.unique().size(0),
                        "pseudo_label_avg_prob_diff_between_1_and_2": diff_top2,
                    }
                )

            else:
                pseudo_y = y_target
                # cross entropy on target
                target_ce_loss = F.cross_entropy(out_t, y_target)

            target_ce_loss_weight = args.target_ce_loss_weight
            loss += target_ce_loss * target_ce_loss_weight
            log["train_target_ce_loss"] = target_ce_loss

            # ****** DA part ******
            if args.da == "cdan":
                transfer_loss_weight = args.transfer_loss_weight
                transfer_loss = CDAN(
                    [feat, softmax_out],
                    self.CDAN_domain_classifier,
                    entropy=None,
                    coeff=None,
                    random_layer=self.random_layer,
                )
                loss += transfer_loss * transfer_loss_weight
                log["train_transfer_loss"] = transfer_loss

            elif args.da == "mmd2":
                mmd_loss_weight = args.mmd_loss_weight
                softmax_source = F.softmax(out_s, dim=1)
                softmax_target = F.softmax(out_t, dim=1)
                input_source = [feat_s, softmax_source]
                input_target = [feat_t, softmax_target]

                kernel_mul = 2.0
                kernel_nums = [2, 5]
                losses_mmd = [
                    mmd2(
                        input_s,
                        input_t,
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_nums[i],
                        fix_sigma=None,
                        ver=2,
                    )
                    for i, (input_s, input_t) in enumerate(zip(input_source, input_target))
                ]
                mmd_loss = sum(losses_mmd) / len(losses_mmd)

                loss += mmd_loss * mmd_loss_weight
                log["train_mmd_loss"] = mmd_loss
            elif args.da == "adversarial":
                adversarial_loss_weight = args.adversarial_loss_weight

                if args.adversarial_coeff == -1:
                    len_dataloader = len(self.trainer.train_dataloader)
                    p = (
                        float(batch_idx + self.trainer.current_epoch * len_dataloader)
                        / self.args.epochs
                        / len_dataloader
                    )
                    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                else:
                    alpha = args.adversarial_coeff

                domain_preds_source = self.domain_classifier(feat_s, alpha)
                labels = torch.zeros(
                    domain_preds_source.size(0), device=self.device, dtype=torch.long
                )
                adversarial_loss = F.cross_entropy(domain_preds_source, labels)

                domain_preds_target = self.domain_classifier(feat_t, alpha)
                labels = torch.ones(
                    domain_preds_target.size(0), device=self.device, dtype=torch.long
                )
                adversarial_loss += F.cross_entropy(domain_preds_target, labels)

                loss += adversarial_loss * adversarial_loss_weight
                log["train_adversarial_loss"] = adversarial_loss

            else:
                z_s = self.contrastive_head(feat_s)
                z_t = self.contrastive_head(feat_t)
                if args.da == "ib":
                    if self.args.use_queue:
                        ib_loss, n_pairs = compute_ib_loss(
                            self,
                            z_s,
                            z_t,
                            y_source,
                            pseudo_y,
                            self.source_queue,
                            self.source_queue_y,
                        )
                    else:
                        ib_loss, n_pairs = compute_ib_loss(self, z_s, z_t, y_source, pseudo_y)
                    loss += ib_loss
                    log["train_ib_loss"] = ib_loss
                    log["n_pairs"] = n_pairs

                if args.da == "vicreg":
                    vicreg_loss = compute_vicreg_loss(self, z_s, z_t, y_source, pseudo_y)
                    loss += vicreg_loss
                    log["train_vicreg_loss"] = vicreg_loss

                if args.da == "simclr":
                    simclr_loss = compute_simclr_loss(self, z_s, z_t, y_source, pseudo_y)
                    loss += simclr_loss
                    log["train_simclr_loss"] = simclr_loss

                # enqueue elements
                if self.args.use_queue:
                    self.dequeue_and_enqueue(z_s, z_t, y_source, pseudo_y)

        # acc1, acc5 = accuracy_at_k(out, torch.cat((y_source, y_target)), top_k=(1, 5))
        # log.update({"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5})
        source_acc1, source_acc5 = accuracy_at_k(out_s, y_source, top_k=(1, 5))
        target_acc1, target_acc5 = accuracy_at_k(out_t, y_target, top_k=(1, 5))
        log.update(
            {
                "train_loss": loss,
                "train_source_acc1": source_acc1,
                "train_source_acc5": source_acc5,
                "train_target_acc1": target_acc1,
                "train_target_acc5": target_acc5,
            }
        )

        return loss, log, feat_s, feat_t

    def training_step(self, batch, batch_idx):
        # non-source free variant
        if len(batch) == 4:
            X_source, y_source, X_target, y_target = batch
            loss, log, feat_s, feat_t = self.multi_domain_training_step(
                X_source, y_source, X_target, y_target, batch_idx
            )
            ret = {
                "loss": loss,
                "y_source": y_source,
                "y_target": y_target,
                "feat_s": feat_s,
                "feat_t": feat_t,
            }
        # source free
        else:
            X, y = batch
            loss, log = self.single_domain_training_step(X, y)
            ret = {"loss": loss, "y": y}

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return ret

    def on_train_epoch_end(self, *args, **kwargs):
        if self.args.plot_feature_visualization:
            self.umap_plot("target_umap", "train")
            self.umap_plot("target_umap_val", "val")

    def umap_plot(self, name, mode):
        data_source = []
        Y_source = []

        data_target = []
        Y_target = []

        self.eval()
        if mode == "train":
            with torch.no_grad():
                for X_source, y_source, X_target, y_target in self.trainer.train_dataloader:
                    X_source = X_source.to(self.device, non_blocking=True)
                    y_source = y_source.to(self.device, non_blocking=True)
                    X_target = X_target.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)

                    feat_s, _ = self(X_source)
                    feat_t, _ = self(X_target)

                    feat_s = gather(feat_s)
                    y_source = gather(y_source)
                    feat_t = gather(feat_t)
                    y_target = gather(y_target)

                    data_source.append(feat_s.cpu())
                    Y_source.append(y_source.cpu())
                    data_target.append(feat_t.cpu())
                    Y_target.append(y_target.cpu())
        else:
            with torch.no_grad():
                for X_target, y_target in self.trainer.val_dataloaders[0]:
                    X_target = X_target.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)

                    feat_t, _ = self(X_target)

                    feat_t = gather(feat_t)
                    y_target = gather(y_target)

                    data_target.append(feat_t.cpu())
                    Y_target.append(y_target.cpu())
        self.set_training()

        if self.trainer.is_global_zero and len(data_target):
            data = torch.cat(data_target, dim=0).numpy()
            y = torch.cat(Y_target, dim=0).numpy()
            domains = ["target"] * data.shape[0]

            if data_source:
                data_source = torch.cat(data_source, dim=0).numpy()
                y_source = torch.cat(Y_source, dim=0).numpy()

                data = np.concatenate((data, data_source))
                y = np.concatenate((y, y_source))
                domains += ["source"] * data_source.shape[0]

            # data = TSNE(n_components=2).fit_transform(data)
            data = umap.UMAP(n_components=2).fit_transform(data)

            # assing to dataframe
            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["y"] = y
            df["domains"] = domains

            plt.figure(figsize=(6, 6))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="y",
                style="domains",
                palette=sns.color_palette("hls", len(np.unique(y))),
                data=df,
                legend="full",
                alpha=0.3,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)

            plt.tight_layout()
            if self.args.wandb:
                wandb.log(
                    {name: wandb.Image(ax)},
                    commit=False,
                )
            else:
                key = str(self.trainer.logger.version)
                plt.savefig(f"umap-{key}.jpg")
            key = str(self.trainer.logger.version)
            try:
                import os

                os.makedirs(f"umaps/{key}")
            except:
                pass
            plt.savefig(f"umaps/{key}/{name}-{self.trainer.current_epoch}.pdf")
            plt.close()
        # torch.distributed.barrier()

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        feat, out = self(X)
        loss = F.cross_entropy(out, target).detach()

        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "outputs": out,
            "targets": target,
            "y_target": target,
            "feat_t": feat,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = weighted_mean(outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outputs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)
