import argparse
import datetime
import math
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import numpy as np
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool

from math import ceil
from layer import *
from torch_dct import dct, idct

import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1
                 , wt_type='haar'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)
        return x
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class DCT(nn.Module):
    def __init__(self, len):
        super().__init__()
        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.

    def forward(self, x_in):
        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.
        return x
class DCT_GRU(nn.Module):
    def __init__(self, dim, GRU_layers):
        super().__init__()
        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.

    def forward(self, x_in):
        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.
        return x

class GNNStack(nn.Module):
    """ The stack layers of GNN.
    """
    def __init__(self):
        super().__init__()
        # TODO: Sparsity Analysis
        gnn_model_type = args.arch
        num_layers = args.num_layers
        groups = args.groups
        kern_size = args.kern_size
        kern_size_mid = args.kern_size_mid
        in_dim = args.in_dim
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        seq_len = args.seq_len

        num_nodes = args.num_channels
        num_classes = args.num_classes

        k_neighs = self.num_nodes = num_nodes
        self.pos_drop = nn.Dropout(p=args.dropout_rate)
        self.num_graphs = args.groups
        self.groups=groups
        self.seq_mid_len = 0
        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += (groups - seq_len % groups)
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)  # return adj
        if seq_len % self.num_graphs:
            pad_size = (self.num_graphs - seq_len % self.num_graphs) / 2
            temp_length = F.pad(torch.randn(3,3,4,2,seq_len), (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
            self.seq_len_new = temp_length.size(-1)
        else:
            self.seq_len_new = seq_len
        GRU_layers = args.GRU_layers

        self.slm = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        self.slm_1 = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        # self.slm = DCT(len=self.seq_len_new)
        # self.slm_1 = DCT(len=self.seq_len_new)

        # This layer is not used (does not participate in forward propagation or backward propagation.);
        # it is solely intended to better initialize the entire network's parameters.
        # self.wt = DepthwiseSeparableConvWithWTConv2d(in_channels=args.num_channels, out_channels=args.num_channels)

        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.

        self.dropout_1 = args.dropout_size[0]  # 0.5
        self.dropout_2 = args.dropout_size[1]  # 0.5
        self.dropout_3 = args.dropout_size[2]  # 0.5
        self.activation = nn.ReLU()  # LeakyReLU()  -----------

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()

    def reset_parameters(self):
        self.tconvs_1.reset_parameters()
        self.gconvs_1.reset_parameters()
        self.bns_1.reset_parameters()
        self.tconvs_2.reset_parameters()
        self.gconvs_2.reset_parameters()
        self.bns_2.reset_parameters()
        self.tconvs_3.reset_parameters()
        self.gconvs_3.reset_parameters()
        self.bns_3.reset_parameters()
        self.tconvs_tmp_1.reset_parameters()
        self.tconvs_tmp_2.reset_parameters()
        self.tconvs_tmp_3.reset_parameters()
        self.linear.reset_parameters()

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1

    def pretrain(self, x_in):
        return x_in

    def forward(self, inputs: Tensor):
        # the critical sections of the code have been treated confidentially.Upon acceptance of the paper,
        # the full code will be provided, and the training weights will be made publicly available.
        # If reviewers require further information, they may contact the authors, and we will provide all details.
        out = inputs
        return out
class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNNStack()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]

        preds, target = self.model.pretrain(data)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNNStack()  # ------
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return pretrain_checkpoint_callback.best_model_path
def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="gpu",  # "auto"
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()

    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # model = model_training.load_from_checkpoint("./lightning_logs/Worms_numLayers3_inDim_54_hiddenDim_108_outDim_216_kernSize_[7, 5, 3]_kernSizeMid_[6, 5, 3]_dropoutSize_[0.4, 0.5, 0.5]14_12_51/epoch=287-step=2592.ckpt")

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    acc_test = str(test_result[0]["test_acc"])
    acc_test = float(acc_test[:5])

    acc_val = str(val_result[0]["test_acc"])
    acc_val = float(acc_val[:5])

    f1_test = str(test_result[0]["test_f1"])
    f1_test = float(f1_test[:5])

    f1_val = str(val_result[0]["test_f1"])
    f1_val = float(f1_val[:5])

    acc_result = {"test": acc_test, "val": acc_val}
    f1_result = {"test": f1_test, "val": f1_val}
    # acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    # f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, acc_result, f1_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='DTFPNet_UCR')
    parser.add_argument('--data_path', type=str, default=r'../dataset/UCR/Worms')
    # parser.add_argument('--data_path', type=str, default=r'../dataset/har')
    # parser.add_argument('--data_path', type=str, default=r'../dataset/UEA/FaceDetection')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=700)  # 190  400  700  1500
    parser.add_argument('--in_dim', type=int, default=54, help='input dimensions of GNN stacks')  # 64
    parser.add_argument('--hidden_dim', type=int, default=108, help='hidden dimensions of GNN stacks')  # 128
    parser.add_argument('--out_dim', type=int, default=216, help='output dimensions of GNN stacks')  # 256
    parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')  # 3
    parser.add_argument('--groups', type=int, default=2, help='the number of time series groups (num_graphs)')  # 2,4,6,8
    parser.add_argument('--kern_size', type=str, default=[7,5,3], help='list of time conv kernel size for each layer')
    parser.add_argument('--kern_size_mid', type=str, default=[6,5,3], help='list of time conv kernel size for each layer')
    parser.add_argument('--dropout_size', type=str, default=[0.4,0.5,0.5], help='list of time conv kernel size for each layer')

    parser.add_argument('--GRU_layers', type=int, default=1, help='layers of GRU')  # 1,4,12,32

    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)    # 16    FaceDetection_4
    parser.add_argument('--train_lr', type=float, default=5e-4)  # 5e-4 1e-4
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False, help='False: without pretraining')

    # TodyNet parameters:
    parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
    parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                        help='validation batch size')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint   ----------------
    run_description = f"{os.path.basename(args.data_path)}_numLayers{args.num_layers}_inDim_{args.in_dim}_"
    run_description += f"hiddenDim_{args.hidden_dim}_outDim_{args.out_dim}_kernSize_{args.kern_size}_kernSizeMid_{args.kern_size_mid}_dropoutSize_{args.dropout_size}"
    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(pretrain_checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # Get dataset characteristics ...
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file...
    text_save_dir = "textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"Ours_{os.path.basename(args.data_path)}_groups_{args.groups}_GRULayers_{args.GRU_layers}_epochs_{args.num_epochs}_train_lr_{args.train_lr}_DCT_GRU" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n')
    f.write('\n')
    f.close()
