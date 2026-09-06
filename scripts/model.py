"""Shared lightweight temporal classifier used for training, evaluation and export."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FallPredictorCNN(nn.Module):
    def __init__(self, input_dim=68, num_classes=2, window_size=16):
        super().__init__()
        if window_size < 4 or window_size % 4:
            raise ValueError("window_size must be divisible by 4 and at least 4")
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (window_size // 4), 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, sequence):
        sequence = sequence.permute(0, 2, 1)
        sequence = self.pool1(self.relu(self.bn1(self.conv1(sequence))))
        sequence = self.pool2(self.relu(self.bn2(self.conv2(sequence))))
        sequence = self.relu(self.fc1(self.flatten(sequence)))
        return self.fc2(self.dropout(sequence))


class FallRiskTimeCNN(nn.Module):
    """Shared lightweight pose backbone with risk classification and time-to-impact heads."""
    def __init__(self, input_dim=68, window_size=32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(128 * (window_size // 4), 64), nn.ReLU(), nn.Dropout(0.5))
        self.risk_head = nn.Linear(64, 1)
        self.time_head = nn.Linear(64, 1)

    def forward(self, sequence):
        value = sequence.permute(0, 2, 1)
        value = self.pool1(self.relu(self.bn1(self.conv1(value))))
        value = self.pool2(self.relu(self.bn2(self.conv2(value))))
        value = self.shared(value)
        return self.risk_head(value).squeeze(1), self.time_head(value).squeeze(1)


class _TemporalBlock(nn.Module):
    """Small residual dilated-convolution block for fixed-length pose sequences."""
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, value):
        residual = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + residual)


class FallPredictorTCN(nn.Module):
    """Candidate temporal convolution network; the input contract remains [N, T, 68]."""
    def __init__(self, input_dim=68, num_classes=3, window_size=32):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[_TemporalBlock(64, dilation) for dilation in (1, 2, 4)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, sequence):
        sequence = sequence.permute(0, 2, 1)
        sequence = self.blocks(self.input(sequence))
        sequence = self.pool(sequence).squeeze(-1)
        return self.classifier(self.dropout(sequence))


class FallPredictorTailTCN(nn.Module):
    """TCN that emphasizes the latest frames instead of averaging a full window."""
    def __init__(self, input_dim=68, num_classes=3, window_size=64, tail_size=16):
        super().__init__()
        self.tail_size = min(tail_size, window_size)
        self.input = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[_TemporalBlock(64, dilation) for dilation in (1, 2, 4)])
        self.attention = nn.Conv1d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, sequence):
        value = sequence.permute(0, 2, 1)
        value = self.blocks(self.input(value))[:, :, -self.tail_size:]
        weights = self.attention(value).squeeze(1).softmax(dim=1).unsqueeze(1)
        value = (value * weights).sum(dim=2)
        return self.classifier(self.dropout(value))


class FallPredictorTCNTE(nn.Module):
    """Causal TCN plus a compact Transformer encoder for online phase prediction."""
    def __init__(self, input_dim=68, num_classes=3, window_size=64):
        super().__init__()
        self.window_size = window_size
        self.input = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[_TemporalBlock(64, dilation) for dilation in (1, 2, 4)])
        self.position = nn.Parameter(torch.zeros(1, window_size, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.2,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, sequence):
        value = self.blocks(self.input(sequence.permute(0, 2, 1))).permute(0, 2, 1)
        if value.shape[1] != self.window_size:
            raise ValueError(f"Expected {self.window_size} frames, received {value.shape[1]}")
        mask = torch.triu(torch.ones(self.window_size, self.window_size, device=value.device, dtype=torch.bool), diagonal=1)
        value = self.encoder(value + self.position, mask=mask)
        return self.classifier(self.dropout(value[:, -1]))


class _GraphTemporalStream(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.register_buffer("base_adjacency", adjacency)
        self.adaptive_adjacency = nn.Parameter(torch.zeros_like(adjacency))
        self.embed = nn.Linear(2, 24)
        self.temporal = nn.Sequential(
            nn.Conv2d(24, 24, (5, 1), padding=(2, 0), groups=24),
            nn.Conv2d(24, 24, 1), nn.BatchNorm2d(24), nn.ReLU(),
        )

    def forward(self, value):
        # value: batch, time, joints, xy-or-velocity
        adjacency = torch.softmax(self.base_adjacency + self.adaptive_adjacency, dim=-1)
        value = self.embed(value)
        value = torch.einsum("ij,btjc->btic", adjacency, value).permute(0, 3, 1, 2)
        return self.temporal(value)


class FallPredictorSTGCNSep(nn.Module):
    """Lightweight skeleton/motion GCN with separable temporal convolutions."""
    def __init__(self, input_dim=68, num_classes=3, window_size=64):
        super().__init__()
        if input_dim != 68:
            raise ValueError("stgcn_sep_v1 requires raw 17-joint coordinate and velocity features")
        edges = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
        adjacency = torch.eye(17)
        for left, right in edges:
            adjacency[left, right] = adjacency[right, left] = 1.0
        self.pose_stream = _GraphTemporalStream(adjacency)
        self.motion_stream = _GraphTemporalStream(adjacency)
        self.residual = nn.Linear(4, 16)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, sequence):
        value = sequence.reshape(sequence.shape[0], sequence.shape[1], 17, 4)
        pose = self.pose_stream(value[..., :2]).mean(dim=(2, 3))
        motion = self.motion_stream(value[..., 2:]).mean(dim=(2, 3))
        residual = self.residual(value).mean(dim=(1, 2))
        return self.classifier(self.dropout(torch.cat((pose, motion, residual), dim=1)))


class FallPredictorThreeBranch(nn.Module):
    """Risk, recoverability and future-pose branches for causal fall prediction.

    The model keeps the deployment input contract [N, T, 68].  The second
    stream derives a small set of interpretable kinematic descriptors from
    coordinates and velocity, while the third stream predicts the next pose
    from the current history.  The auxiliary tasks make the shared embedding
    distinguish sustained descent from a motion that stabilizes again.
    """
    def __init__(self, input_dim=68, num_classes=3, window_size=64, num_action_classes=10):
        super().__init__()
        if input_dim != 68:
            raise ValueError("three_branch_v1 requires raw 17-joint coordinates and velocity")
        self.window_size = window_size
        self.pose = nn.Sequential(
            nn.Conv1d(68, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2, groups=4), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.motion = nn.Sequential(
            nn.Conv1d(8, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2, groups=4), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.pose_pool = nn.AdaptiveAvgPool1d(1)
        self.motion_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.25))
        self.risk_head = nn.Linear(64, num_classes)
        self.recovery_head = nn.Linear(32, 1)
        self.forecast_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 34))
        self.action_head = nn.Linear(64, num_action_classes)

    @staticmethod
    def kinematic_features(sequence):
        points = sequence[..., :34].reshape(sequence.shape[0], sequence.shape[1], 17, 2)
        velocity = sequence[..., 34:].reshape(sequence.shape[0], sequence.shape[1], 17, 2)
        shoulder = (points[:, :, 5] + points[:, :, 6]) / 2
        hip = (points[:, :, 11] + points[:, :, 12]) / 2
        ankle = (points[:, :, 15] + points[:, :, 16]) / 2
        torso = shoulder - hip
        angle = torch.atan2(torso[..., 1], torso[..., 0])
        hip_velocity = (velocity[:, :, 11, 1] + velocity[:, :, 12, 1]) / 2
        hip_acceleration = torch.diff(hip_velocity, dim=1, prepend=hip_velocity[:, :1])
        gap = hip[..., 1] - ankle[..., 1]
        gap_delta = torch.diff(gap, dim=1, prepend=gap[:, :1])
        visible = (points.abs().sum(dim=-1) > 1e-5).float().mean(dim=-1)
        return torch.stack((angle, hip[..., 1], hip_velocity, hip_acceleration,
                            (points[:, :, 15, 0] - points[:, :, 16, 0]).abs(),
                            gap, gap_delta, visible), dim=-1)

    def forward(self, sequence):
        pose_stream = self.pose(sequence.permute(0, 2, 1))
        motion_stream = self.motion(self.kinematic_features(sequence).permute(0, 2, 1))
        pose_embedding = self.pose_pool(pose_stream).squeeze(-1)
        motion_embedding = self.motion_pool(motion_stream).squeeze(-1)
        fused = self.fusion(torch.cat((pose_embedding, motion_embedding), dim=1))
        return (self.risk_head(fused), self.recovery_head(motion_embedding).squeeze(1),
                self.forecast_head(motion_embedding), self.action_head(fused))


class PaperMultiHorizonTeacher(nn.Module):
    """Causal pose Transformer for the paper-only multi-horizon task.

    This model intentionally has no K230 export constraint. It consumes only
    normalized 2D pose coordinates and velocities, and returns three nested
    future-risk logits plus a bounded time-to-impact estimate.
    """
    def __init__(self, input_dim=68, window_size=64, hidden_dim=128, layers=3):
        super().__init__()
        self.window_size = window_size
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.motion_stem = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=4),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(hidden_dim, 3))
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, sequence):
        if sequence.shape[1] != self.window_size:
            raise ValueError(f"Expected {self.window_size} frames, received {sequence.shape[1]}")
        value = self.input(sequence)
        value = value + self.motion_stem(value.transpose(1, 2)).transpose(1, 2)
        mask = torch.triu(torch.ones(self.window_size, self.window_size, device=value.device, dtype=torch.bool), diagonal=1)
        value = self.encoder(value + self.position, mask=mask)
        embedding = self.norm(value[:, -1])
        return self.risk_head(embedding), torch.sigmoid(self.time_head(embedding)).squeeze(1)


class PaperMultiHorizonRecoveryTeacher(PaperMultiHorizonTeacher):
    """Paper teacher with an auxiliary probability that an action will recover."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, sequence):
        value = self.input(sequence)
        value = value + self.motion_stem(value.transpose(1, 2)).transpose(1, 2)
        mask = torch.triu(torch.ones(self.window_size, self.window_size, device=value.device, dtype=torch.bool), diagonal=1)
        value = self.encoder(value + self.position, mask=mask)
        embedding = self.norm(value[:, -1])
        return (self.risk_head(embedding), torch.sigmoid(self.time_head(embedding)).squeeze(1),
                self.recovery_head(embedding).squeeze(1))


class PaperMultiHorizonActionTeacher(PaperMultiHorizonTeacher):
    """Paper teacher with an action head for intentional normal movements."""
    def __init__(self, *args, action_classes=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15),
                                         nn.Linear(64, action_classes))

    def forward(self, sequence):
        value = self.input(sequence)
        value = value + self.motion_stem(value.transpose(1, 2)).transpose(1, 2)
        mask = torch.triu(torch.ones(self.window_size, self.window_size, device=value.device, dtype=torch.bool), diagonal=1)
        value = self.encoder(value + self.position, mask=mask)
        embedding = self.norm(value[:, -1])
        return (self.risk_head(embedding), torch.sigmoid(self.time_head(embedding)).squeeze(1),
                self.action_head(embedding))


class PaperTimeBinActionTeacher(PaperMultiHorizonActionTeacher):
    """Causal teacher that predicts an ordered time-to-impact phase.

    Classes are normal, more than 1 second, 0.5-1 second, 0.25-0.5 second,
    and within 0.25 second.  Cumulative class probabilities are converted to
    the three paper risk horizons during evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(128, 5))


class ActionContextCNN(nn.Module):
    """Small 32-frame pose classifier used only to identify intentional/recovered actions."""
    def __init__(self, input_dim=68, num_classes=7, window_size=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * (window_size // 4), 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, sequence):
        return self.classifier(self.features(sequence.permute(0, 2, 1)))


class CausalDilatedBlock(nn.Module):
    """One residual TCN block: two causal dilated convolutions (k=3).

    Padding is trimmed on the right so every output position only sees the
    current and previous frames (strictly causal), matching the teacher's
    causal attention mask.
    """
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        pad = 2 * dilation  # (k-1)*d for k=3
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        t = x.size(2)
        out = self.drop(F.gelu(self.norm1(self.conv1(x)))[:, :, :t])
        out = self.drop(F.gelu(self.norm2(self.conv2(out)))[:, :, :t])
        return x + out


class TCNTEBaseline(nn.Module):
    """TCN + Transformer encoder baseline for the fair-reproduction study (paper §7.1.4).

    Backbone-only replacement of the paper teacher: a stack of causal dilated
    TCN blocks captures local temporal dynamics, then the same causal
    Transformer encoder aggregates long-range context. Heads, losses, data,
    split, budget and evaluation protocol are IDENTICAL to the paper teacher;
    the backbone is the only variable. Architecture follows the
    TCN+Transformer-encoder design of Yu et al., 2025 (TCNTE), reimplemented
    from the paper description (no official code available).
    """
    def __init__(self, input_dim=68, window_size=64, hidden_dim=128, layers=3, action_classes=8):
        super().__init__()
        self.window_size = window_size
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        # TCN stem: dilations 1,2,4,8 -> receptive field ~ 33 frames (<64, causal)
        self.tcn = nn.Sequential(*[CausalDilatedBlock(hidden_dim, d) for d in (1, 2, 4, 8)])
        self.position = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(hidden_dim, 3))
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(0.15),
                                          nn.Linear(64, action_classes))

    def forward(self, sequence):
        if sequence.shape[1] != self.window_size:
            raise ValueError(f"Expected {self.window_size} frames, received {sequence.shape[1]}")
        value = self.input(sequence)
        value = value + self.tcn(value.transpose(1, 2)).transpose(1, 2)
        mask = torch.triu(torch.ones(self.window_size, self.window_size, device=value.device, dtype=torch.bool), diagonal=1)
        value = self.encoder(value + self.position, mask=mask)
        embedding = self.norm(value[:, -1])
        return (self.risk_head(embedding), torch.sigmoid(self.time_head(embedding)).squeeze(1),
                self.action_head(embedding))


class PureTCNBaseline(nn.Module):
    """Pure causal TCN baseline with the paper task heads and no attention."""

    def __init__(self, input_dim=68, window_size=64, hidden_dim=128, action_classes=8):
        super().__init__()
        self.window_size = window_size
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.tcn = nn.Sequential(*[CausalDilatedBlock(hidden_dim, d) for d in (1, 2, 4, 8)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(hidden_dim, 3))
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                       nn.Linear(hidden_dim // 2, 1))
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(0.15),
                                         nn.Linear(64, action_classes))

    def forward(self, sequence):
        if sequence.shape[1] != self.window_size:
            raise ValueError(f"Expected {self.window_size} frames, received {sequence.shape[1]}")
        value = self.input(sequence).transpose(1, 2)
        embedding = self.norm(self.tcn(value).transpose(1, 2)[:, -1])
        return (self.risk_head(embedding), torch.sigmoid(self.time_head(embedding)).squeeze(1),
                self.action_head(embedding))


def build_predictor(architecture="cnn_v1", input_dim=68, num_classes=3, window_size=32, **kwargs):
    """Create a predictor from the architecture name stored in its checkpoint."""
    if architecture in {"cnn", "cnn_v1", None}:
        return FallPredictorCNN(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    if architecture == "tcn_v1":
        return FallPredictorTCN(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    if architecture == "tcn_tail_v1":
        return FallPredictorTailTCN(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    if architecture == "tcn_te_v1":
        return FallPredictorTCNTE(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    if architecture == "stgcn_sep_v1":
        return FallPredictorSTGCNSep(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    if architecture == "three_branch_v1":
        return FallPredictorThreeBranch(input_dim=input_dim, num_classes=num_classes, window_size=window_size,
                                        num_action_classes=kwargs.get("num_action_classes", 10))
    if architecture == "action_context_v1":
        return ActionContextCNN(input_dim=input_dim, num_classes=num_classes, window_size=window_size)
    raise RuntimeError(f"Unsupported predictor architecture: {architecture}")
