
import torch
import torch.nn as nn
from torchvision.ops.misc import Permute

from ._network import _Network
from .modules import LambdaModule

class TNet(nn.Module):
    def __init__(self, k=3, batch_norm=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self. k = k
        self.permute = Permute([0, 2, 1])
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv1d(128 ,1024, 1),
            nn.BatchNorm1d(1024) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
        )
        # agnostic to number of points
        self.maxpool = LambdaModule(lambda x: torch.max(x, 2, keepdim=True)[0])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
        )
        self.pose = nn.Linear(256, k*k)

        self.identity_vector = nn.Parameter(torch.eye(k).flatten(), requires_grad=False)

    def forward(self, x: torch.Tensor):
        
        # x = self.permute(x)
        # B C L
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        x = self.pose(x)
        x += self.identity_vector
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(_Network):
    def __init__(self, batch_norm=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.permute = Permute([0, 2, 1])

        self.input_tnet = TNet(3)

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
        )

        self.feature_tnet = TNet(64)

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024) if batch_norm else nn.Identity(),
        )

        self.maxpool = LambdaModule(lambda x: torch.max(x, 2, keepdim=True)[0])

        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512) if batch_norm else nn.Identity(),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256) if batch_norm else nn.Identity(),
        #     nn.LeakyReLU(),
        # )

        # self.pose = nn.Linear(256, 12)


    def forward(self, x: torch.Tensor):

        # B L C or B L 3
        
        num_points = x.size(1)

        x = self.permute(x)

        T1 = self.input_tnet(x)
        x = self.permute(torch.bmm(self.permute(x), T1))

        x = self.conv1(x)

        T2 = self.feature_tnet(x)
        x = self.permute(torch.bmm(self.permute(x), T2))

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = x.squeeze(-1)

        # x = self.fc(x)

        # x = self.pose(x)

        # x = x.view(-1, 3, 4)

        return x

class PoseNet(_Network):
    def __init__(self, batch_norm=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.source_net = PointNet()
        self.target_net = PointNet()

        DIMS = 1024 * 2
        self.fc = nn.Sequential(
            nn.Linear(DIMS, DIMS),
            nn.BatchNorm1d(DIMS) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(DIMS, DIMS//2),
            nn.BatchNorm1d(DIMS//2) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(DIMS//2, DIMS//4),
            nn.BatchNorm1d(DIMS//4) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
        )

        self.pose = nn.Linear(DIMS//4, 12)
    
    def forward(self, s, t):

        s = self.source_net(s)
        t = self.target_net(t)

        x = torch.cat((s, t), dim=-1)
        x = self.fc(x)

        x = self.pose(x)

        x = x.view(-1, 3, 4)

        return x      
