# -*- coding: utf-8 -*-
"""
This is a program for reproducing DEC.
(Unsupervised Deep Embedding for Clustering Analysis - https://proceedings.mlr.press/v48/xieb16.pdf)
Author: Guanbao Liang
License: BSD 2 clause
"""

import datetime
import time
import sys
sys.path.append('./')

import numpy as np
import torch

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

from DataLoader import base_transforms
from representation import StackedAutoEncoder
from clustering import Clustering, Loss
from data.cv_dataset import load_torchvision_dataset
from utils.init_env import init, init_optimizer, init_backbone
from utils.evaluation import eva
from utils.log_save import save_param, save_train_details, save_eva_details, save_model


class DEC(nn.Module):
    """
    This is the entire DEC components.
    (Unsupervised Deep Embedding for Clustering Analysis - https://proceedings.mlr.press/v48/xieb16.pdf)

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    n_clusters : int
        The number of clusters.
    n_init : int
        The number of kmeans executions.
    max_iter : int
        The max iteration of Kmeans algorithm.
    tol : float
        The tolerance of Kmeans algorithm.
    alpha : float
        The parameter in Student's t-distribution which defaults to 1.0.
    pretrain : bool
        Whether to choice the pretrain mode.
    pretrain_path : str
        The place of the pretrain model.
    device_name : str
        The network will train on the device.
    """

    def __init__(
        self,
        in_dim=784,
        n_clusters=10,
        n_init=20,
        max_iter=20000,
        tol=0.001,
        alpha=1.0,
        pretrain=True,
        pretrain_path=None,
        device_name="cuda",
    ):
        super(DEC, self).__init__()
        self.in_dim = in_dim
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.pretrain = pretrain
        self.pretrain_path = pretrain_path
        self.device = device_name

    def init_pretrain(self, x):
        """
        Funtion that initializes the stacked autoencoder and clustering layer.

        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        None
        """
        state_dict = torch.load(self.pretrain_path)
        representation = StackedAutoEncoder(self.in_dim).autoencoder
        representation.load_state_dict(state_dict["net_model"])
        self.representaion = representation[0].to(self.device)

        self.representaion.eval()
        with torch.no_grad():
            x = self.representaion(x.to(self.device)).cpu()

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.labels = torch.tensor(
            kmeans.fit_predict(x.cpu().numpy()), dtype=torch.float, requires_grad=True
        )
        cluster_centers = kmeans.cluster_centers_
        self.clustering = Clustering(
            x.shape[1], self.n_clusters, self.alpha, cluster_centers
        ).to(self.device)
        self.built = True

    def forward(self, x):
        """
        Forward Propagation.

        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        c : torch.Tensor
            Clustering assignments.
        """
        feat = self.representaion(x)
        c = self.clustering(feat)
        return c

    def predict(self, x):
        """
        Function that calculates the probability of assigning sample i to cluster j

        Parameters
        ----------
        x : torch.Tensor
            The node features matrix.

        Returns
        -------
        out : torch.Tensor
            The clustering assignment.
        """
        with torch.no_grad():
            feat = self.representaion(x)
            c = self.clustering(feat)
            return torch.argmax(c, dim=1)


args = init(config_file=["configs/base.yaml", "configs/DEC.yaml"])
args.img_size_at = (28, 28)
save_param(log_dir=args.log_dir, param_dict=vars(args))
writer = SummaryWriter(log_dir=args.log_dir)

trans = base_transforms(resize=args.img_size_at)
dataset = load_torchvision_dataset(
    args.dataset_name,
    args.dataset_dir,
    train=True,
    transform=trans,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=False,
)

dataset_eval = load_torchvision_dataset(
    args.dataset_name,
    args.dataset_dir,
    train=True,
    transform=trans,
)
dataloader_eval = DataLoader(
    dataset=dataset_eval,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False,
)

train_X = torch.cat([data for data, label in dataset], dim=0)
train_X = train_X.reshape(len(dataset), -1)
in_dim = train_X.shape[1]

criterion = Loss().to(args.device)
model = DEC(
    in_dim=in_dim,
    n_clusters=args.class_num,
    pretrain_path=args.pretrain_path,
    device_name=args.device,
).to(args.device)

model.init_pretrain(train_X)
optimizer = init_optimizer(
    optimizer_name=args.optimizer,
    lr=args.lr,
    weight_decay=args.weight_decay,
    params=model.parameters(),
    sgd_momentum=args.sgd_momentum,
)
for epoch in range(args.start_epoch, args.epochs + 1):
    model.train()
    total_examples = 0
    total_loss = 0
    start_time = time.time()
    for step, (data, _) in enumerate(dataloader):
        small_start_time = time.time()
        data = data.to(args.device)
        data = data.reshape(data.shape[0], -1)
        q = model(data)
        with torch.no_grad():
            weight = q**2 / q.sum(dim=0)
            p = (weight.T / weight.sum(dim=1)).T

        loss = criterion(q.log(), p)
        total_loss += loss.item() * data.size(0)
        total_examples += data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        small_end_time = time.time()
        train_details = "[Epoch : {:04d}/{:04d}] ,[Mini-batch : {:04d}/{:04d}], [total_loss : {:.7f}], [time : {:.3f}s]".format(
            epoch - args.start_epoch + 1,
            args.epochs - args.start_epoch + 1,
            step + 1,
            len(dataloader),
            loss.item(),
            small_end_time - small_start_time,
        )
        if args.verbose:
            print(train_details)
        save_train_details(args.log_dir, train_details + "\n")
    avg_loss = total_loss / total_examples
    writer.add_scalar("Loss/train", avg_loss, epoch)
    end_time = time.time()
    train_details = "[Epoch : {:04d}/{:04d}], [loss : {:.7f}], [time : {:.3f}s]".format(
        epoch, args.epochs, avg_loss, end_time - start_time
    )
    if args.verbose:
        print(train_details)
    save_train_details(args.log_dir, train_details + "\n\n")

    if epoch % args.eval_step == 0:
        if args.verbose:
            print("\nEvaluate the model is starting...")
        model.eval()
        feature_vector = []
        labels_vector = []
        with torch.no_grad():
            for step, (x, y) in enumerate(dataloader_eval):
                x = x.to(args.device)
                x = x.view(x.size(0), -1)
                c = model.predict(x)
                feature_vector.extend(c.cpu().numpy())
                labels_vector.extend(y.cpu().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        acc, f1, nmi, ari = eva(labels_vector, feature_vector)

        writer.add_scalar("Eval/acc", acc, epoch)
        writer.add_scalar("Eval/f1", f1, epoch)
        writer.add_scalar("Eval/nmi", nmi, epoch)
        writer.add_scalar("Eval/ari", ari, epoch)
        eval_details = "[Epoch : {:04d}/{:04d}], [acc : {:.4f}%], [f1 : {:.4f}%], [nmi : {:.4f}%], [ari : {:.4f}%]".format(
            epoch,
            args.epochs - args.start_epoch + 1,
            acc * 100,
            f1 * 100,
            nmi * 100,
            ari * 100,
        )
        save_eva_details(args.log_dir, eval_details + "\n\n")
        if args.verbose:
            print("Evaluate the model is over...\n")
            print(eval_details)

    if epoch % args.save_step == 0:
        if args.verbose:
            print("\nSave the model is starting...")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_size = 0
        for param in model.parameters():
            if param.requires_grad:
                model_size += param.data.nelement()
        ckpt = {
            "current_time": current_time,
            "args": vars(args),
            "iteration_num": epoch,
            "model_size": "{:.2f} MB".format(model_size / 1024 / 1024),
            "net_model": model.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer": optimizer.state_dict(),
        }
        save_model(args.model_dir, ckpt, epoch)
        if args.verbose:
            print("Save the model is over...\n")
