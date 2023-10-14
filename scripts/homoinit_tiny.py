import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq._impl.homo_misc import VariationalHidDropout2d
from tiny_imagenet import TinyImageNet

torch.manual_seed(1)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--lam_skip', type=float, default=1.0)
parser.add_argument('--lam_ent', type=float, default=0.0)
parser.add_argument('--b', type=float, default=0.0)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res', 'skip'])
parser.add_argument('--nepochs', type=int, default=300)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--etol', type=float, default=1e-3)
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--freq', type=int, default=20)
parser.add_argument('--init_scheduler', type=int, default=1)
parser.add_argument('--lr_init', type=float, default=0.05)

parser.add_argument('--use_init_bp', type=eval, default=True, choices=[True, False])
parser.add_argument('--use_skip_bp', type=eval, default=False, choices=[True, False])
parser.add_argument('--semi_grad', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=400)
parser.add_argument('--warmup', type=int, default=-1)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

from torchdiffeq import homoint_adjoint
from torchdiffeq import homoint

def entropy_regularizer(logits):
    p = F.softmax(logits, 1)
    return (p*torch.log(p)).sum(1).mean()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(8, dim), dim)

def create_init(x):
    z = torch.ones_like(x)
    z[:, ::2] = -z[:, ::2]
    return z

class InitLayer(nn.Module):
    def __init__(self, dim, kernel=1):
        super().__init__()
        if kernel == 1:
            self.init = nn.Sequential(
                conv1x1(1, dim),
                norm(dim)
            )
        elif kernel == 3:
            self.init = nn.Sequential(
                conv3x3(1, dim),
                norm(dim)
            )
    def forward(self, x):
        n, c, h, w = x.size()
        x = torch.ones(n, 1, h, w, device=x.device)
        return self.init(x)
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        # self._layer = module(
        #     dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
        #     bias=bias
        # )
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        # tt = torch.ones_like(x[:, :1, :, :]) * t
        # ttx = torch.cat([tt, x], 1)
        ttx = x
        return self._layer(ttx)


class Homofunc(nn.Module):

    def __init__(self, dim):
        super(Homofunc, self).__init__()
        # self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.drop = VariationalHidDropout2d(0.15)
        self.nfe = 0

    def forward(self, t, z, x):
        self.nfe += 1
        # x = self.norm1(x)
        # x = self.relu(x)
        out = z
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out) + x
        out = self.drop(self.conv2(t, out))
        out = self.norm3(out)
        return out

    def reset(self, bsz, d, H, W, device):
        self.drop.reset_mask(bsz=bsz, d=d, H=H, W=W, device=device)

class HomoBlock(nn.Module):

    def __init__(self, odefunc, hidden_dim, use_adjoint):
        super(HomoBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.use_adjoint = use_adjoint

    def forward(self, x, z=None, init_layer=None):
        n, c, h, w = x.size()
        self.odefunc.reset(n, c, h, w, x.device)
        if init_layer is None and z is None:
            z0 = create_init(x) # torch.zeros_linke(x)
        elif z is None:
            z0 = init_layer(x).detach()
        else:
            z0 = z
        self.integration_time = self.integration_time.type_as(z0)
        if not self.use_adjoint:
            out = homoint(self.odefunc, z0, x, self.integration_time, rtol=args.tol, atol=args.tol)
        else:
            out = homoint_adjoint(self.odefunc, z0, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    def adjoint(self, use_adjoint=True):
        self.use_adjoint = use_adjoint

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_tiny_imagenet_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(64, padding=8),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = DataLoader(
        TinyImageNet(root='./data/tiny-imagenet', train=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        TinyImageNet(root='./data/tiny-imagenet', train=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        TinyImageNet(root='./data/tiny-imagenet', train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def get_correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        if len(topk) == 1:
            res.append(torch.zeros_like(res[-1]))
        return res

def accuracy(model, dataset_loader, init_model=None):
    total_correct1 = 0
    total_correct5 = 0
    for x, y in dataset_loader:
        x = x.to(device)
        output = model(x, init_layer=init_model)[0].cpu()
        correct1, correct5 = get_correct(output, y, topk=(1,5))
        total_correct1 += correct1
        total_correct5 += correct5
        # predicted_class = np.argmax(model(x, init_layer=init_model)[0].cpu().detach().numpy(), axis=1)
    return total_correct1 / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

class HomoSkip(nn.Module):
    def __init__(self, hidden_dim, sep=True):

        super().__init__()
        self.sep = sep
        if sep:
            self.condition_layers = nn.Sequential(*[
                nn.Conv2d(3, hidden_dim, 3, 1, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

            self.initilize_layers = nn.Sequential(*[
                nn.Conv2d(3, hidden_dim, 3, 1, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        else:
            self.downsampling_layers = nn.Sequential(*[
                nn.Conv2d(3, hidden_dim, 3, 1, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

            self.condition_layers = nn.Sequential(*[
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

            self.initilize_layers = nn.Sequential(*[
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
    def forward(self, x, mode='c'):
        out = x if self.sep else self.downsampling_layers(x)
        if mode == 'c':
            cond = self.condition_layers(out)
            with torch.no_grad():
                init = self.initilize_layers(out)
        elif mode == 'i':
            with torch.no_grad():
                cond = self.condition_layers(out)
            init = self.initilize_layers(out)
        else:
            cond = self.condition_layers(out)
            init = self.initilize_layers(out)


        return cond, init
class HomoNet(nn.Module):

    def __init__(self, hidden_dim=64, final_dim=256, use_init_bp=True, use_skip_bp=False, use_adjoint=False, semi_grad=True):
        super().__init__()
        self.skip = args.downsampling_method == 'skip'
        if args.downsampling_method == 'conv':
            self.downsampling_layers = nn.Sequential(*[
                nn.Conv2d(3, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        elif args.downsampling_method == 'res':
            self.downsampling_layers = nn.Sequential(*[
                nn.Conv2d(3, hidden_dim, 3, 1),
                ResBlock(hidden_dim, hidden_dim, stride=2, downsample=conv1x1(hidden_dim, hidden_dim, 2)),
                ResBlock(hidden_dim, hidden_dim, stride=2, downsample=conv1x1(hidden_dim, hidden_dim, 2)),
            ])
        elif args.downsampling_method == 'skip':
            self.downsampling_layers = HomoSkip(hidden_dim, sep=True)
            self.use_init_bp = use_init_bp
            self.use_skip_bp = use_skip_bp
            self.semi_grad = semi_grad

        self.is_odenet = args.network == 'odenet'
        self.feature_layers = HomoBlock(Homofunc(hidden_dim), hidden_dim=hidden_dim, use_adjoint=use_adjoint) if self.is_odenet else nn.Sequential(*[ResBlock(hidden_dim, hidden_dim) for _ in range(6)])
        # self.fc_layers = FinalLayer(hidden_dim, hidden_dim//4, 4, final_dim, 200)
        self.fc_layers = nn.Sequential(*[norm(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, final_dim, 1, 1), nn.BatchNorm2d(final_dim),
                     nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(final_dim, 200)])

    def forward(self, x, mode='c', init_layer=None):
        if self.skip:
            cond, init = self.downsampling_layers(x, mode)
            if mode == 'i':
                with torch.no_grad():
                    if self.training:
                        out = self.feature_layers(cond, init.clone() if self.use_init_bp else init.clone().detach())
                    else:
                        out = self.feature_layers(cond, init)
                sol = out.detach().clone()
                with torch.no_grad():
                    out = self.fc_layers(out)
            else:
                if self.training:
                    out = self.feature_layers(cond, init.clone() if self.use_init_bp else init.clone().detach())
                else:
                    out = self.feature_layers(cond, init)
                sol = out.detach().clone()
                out = self.fc_layers(out)
        else:
            out = self.downsampling_layers(x)
            fea = out.detach().clone()
            out = self.feature_layers(out) if init_layer is None else self.feature_layers(out, init_layer=init_layer)
            sol = out.detach().clone()
            out = self.fc_layers(out)

        return out, fea, sol

    def set_adjoint(self):
        if self.is_odenet:
            self.feature_layers.adjoint(True)

    def unset_adjoint(self):
        if self.is_odenet:
            self.feature_layers.adjoint(False)

    def flip_adjoint(self):
        if self.is_odenet:
            self.feature_layers.adjoint(not self.feature_layers.use_adjoint)

class FinalLayer(nn.Module):

    def __init__(self, hidden_dim, dim, expansion, final_dim, cls_dim):
        super().__init__()
        self.norm1 = norm(hidden_dim)
        self.conv = nn.Sequential(*[nn.Conv2d(hidden_dim, dim, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                                     nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                                     nn.Conv2d(dim, dim*expansion, 1, 1), nn.BatchNorm2d(dim*expansion)])
        self.fc_layers = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, final_dim, 1, 1), nn.BatchNorm2d(final_dim),
                             nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(final_dim, cls_dim)])
    def forward(self, x):
        x = self.norm1(x)
        out = self.conv(x)
        out += x
        return self.fc_layers(out)

if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = HomoNet(hidden_dim=128, final_dim=512, use_init_bp=args.use_init_bp, use_skip_bp=args.use_skip_bp, use_adjoint=args.adjoint, semi_grad=args.semi_grad).to(device)
    init_model = InitLayer(dim=128, kernel=1).to(device)

    logger.info(model)
    logger.info('Number of parameters: {} K'.format(count_parameters(model) // 1000))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_tiny_imagenet_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_optimizer = torch.optim.SGD(init_model.parameters(), lr=args.lr_init)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs * batches_per_epoch, eta_min=1e-6)
    init_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(init_optimizer, T_max=args.nepochs * batches_per_epoch // args.init_scheduler, eta_min=1e-6)

    correct = 0
    total_loss = 0
    total_extra_loss = 0
    total_entropy = 0

    for itr in range(args.nepochs * batches_per_epoch):
        if itr == args.warmup:
            model.flip_adjoint()
        mode = 's'
        if itr % 500 < 0:
            mode = 'i'
        model.train()
        optimizer.zero_grad()
        init_optimizer.zero_grad()

        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits, fea, sol = model(x, mode, init_model)
        loss = torch.abs(criterion(logits, y) - args.b) + args.b
        extra_loss = torch.nn.functional.smooth_l1_loss(init_model(fea), sol)
        entropy = entropy_regularizer(logits)
        total_loss += loss.item()
        total_extra_loss += extra_loss.item()
        total_entropy += entropy.item()
        loss += args.lam_ent * entropy

        tmp1, _ = get_correct(logits.cpu(), y.cpu(), topk=(1,))
        correct += tmp1

        if model.is_odenet:
            nfe_forward = model.feature_layers.nfe
            model.feature_layers.nfe = 0

        loss.backward()
        optimizer.step()
        scheduler.step()
        if itr % args.freq == 0:
            extra_loss.backward()
            init_optimizer.step()
            init_scheduler.step()

        if model.is_odenet:
            nfe_backward = model.feature_layers.nfe
            model.feature_layers.nfe = 0

        batch_time_meter.update(time.time() - end)
        if model.is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == batches_per_epoch - 1:
            model.eval()
            with torch.no_grad():
                total_loss /= batches_per_epoch
                total_entropy /= batches_per_epoch
                total_extra_loss /= batches_per_epoch
                train_acc = correct / len(train_loader.dataset)
                val_acc = accuracy(model, test_loader, init_model)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Train Loss {:.4f} | Extra Loss {:.4f} | Ent Loss {:.4f} | Test Acc {:.4f} | Best Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc.item(), total_loss, total_extra_loss, total_entropy, val_acc.item(), best_acc.item()
                    )
                )
                model.feature_layers.nfe = 0
            correct = 0
            total_loss = 0
            total_extra_loss = 0
            total_entropy = 0