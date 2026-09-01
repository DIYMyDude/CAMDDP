#========================= model_4class.py =========================
#This file contains the model representation and supporting functions for CAMDDP four-class classification.
#==============================================================

import torch
import torch.nn as nn
import torchvision.models as models

# ==================================================
# CONFIGURATION
# ==================================================
NUM_CLASSES = 4 #DO NOT CHANGE

from config import (
    VISUAL_FEATURE_DIM,
    AUDIO_FILTERS,
    AUDIO_KERNEL_SIZE,
    AUDIO_USE_BATCHNORM,
    AUDIO_ACTIVATION,
    AUDIO_POOL_TYPE,
    AUDIO_POOL_SIZE,
    AUDIO_USE_GAUSSIAN_NOISE,
    AUDIO_GAUSSIAN_NOISE_STD,
    AUDIO_USE_GAP,
    AUDIO_EMBEDDING_DIM,
    AUDIO_USE_DROPOUT,
    AUDIO_DROPOUT_RATE,
    AUDIO_MFCC_COEFFICIENTS,
    FUSION_USE_HIDDEN_LAYER,
    FUSION_HIDDEN_DIM
)

# ==================================================
# GAUSSIAN NOISE
# ==================================================

class GaussianNoise(nn.Module):

    def __init__(
        self,
        std=0.05
    ):

        super().__init__()

        self.std = std

    def forward(
        self,
        x
    ):

        if self.training:

            noise = torch.randn_like(
                x
            ) * self.std

            return x + noise

        return x


# ==================================================
# AUDIO BRANCH
# ==================================================

class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        layers = []

        if AUDIO_USE_GAUSSIAN_NOISE:

            layers.append(
                GaussianNoise(
                    AUDIO_GAUSSIAN_NOISE_STD
                )
            )

        in_channels = 1

        for out_channels in AUDIO_FILTERS:

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=AUDIO_KERNEL_SIZE,
                    padding=AUDIO_KERNEL_SIZE // 2
                )
            )

            if AUDIO_USE_BATCHNORM:

                layers.append(
                    nn.BatchNorm2d(
                        out_channels
                    )
                )

            if AUDIO_ACTIVATION.lower() == "relu":

                layers.append(
                    nn.ReLU(
                        inplace=True
                    )
                )

            if AUDIO_POOL_TYPE.lower() == "max":

                layers.append(
                    nn.MaxPool2d(
                        AUDIO_POOL_SIZE
                    )
                )

            else:

                layers.append(
                    nn.AvgPool2d(
                        AUDIO_POOL_SIZE
                    )
                )

            in_channels = out_channels

        self.conv_stack = nn.Sequential(
            *layers
        )

        if AUDIO_USE_GAP:

            self.pool = nn.AdaptiveAvgPool2d(
                (1, 1)
            )

            fc_input_size = AUDIO_FILTERS[-1]

        else:

            self.pool = nn.Flatten()

            fc_input_size = (
                AUDIO_FILTERS[-1]
                * 16
                * 50
            )

        self.fc = nn.Linear(
            fc_input_size,
            AUDIO_EMBEDDING_DIM
        )

        self.audio_confidence = nn.Sequential(
            nn.Linear(
                AUDIO_EMBEDDING_DIM,
                1
            ),
            nn.Sigmoid()
        )

        if AUDIO_USE_DROPOUT:

            self.dropout = nn.Dropout(
                AUDIO_DROPOUT_RATE
            )

        else:

            self.dropout = nn.Identity()

    def forward(
        self,
        x
    ):

        x = self.conv_stack(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc(x)

        x = self.dropout(x)

        confidence = self.audio_confidence(
            x
        )

        return (
            x,
            confidence
        )


# ==================================================
# VISUAL BRANCH
# ==================================================

class VisualEfficientNetB0(
    nn.Module
):

    def __init__(self):

        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.gap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.visual_confidence = nn.Sequential(
            nn.Linear(
                VISUAL_FEATURE_DIM,
                1
            ),
            nn.Sigmoid()
        )
        

    def forward(
        self,
        frames
    ):

        batch_size = frames.size(0)

        num_frames = frames.size(1)

        frames = frames.view(
            batch_size * num_frames,
            3,
            224,
            224
        )

        features = self.feature_extractor(
            frames
        )

        features = self.gap(
            features
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

        features = features.view(
            batch_size,
            num_frames,
            VISUAL_FEATURE_DIM
        )

        features = torch.mean(
            features,
            dim=1
        )

        confidence = self.visual_confidence(
            features
        )

        return (
            features,
            confidence
        )


# ==================================================
# MULTIMODAL MODEL
# ==================================================

class Model(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.audio_branch = AudioCNN()

        self.visual_branch = (
            VisualEfficientNetB0()
        )

        fusion_dim = (
            AUDIO_EMBEDDING_DIM
            + VISUAL_FEATURE_DIM
            + 2
        )

        if FUSION_USE_HIDDEN_LAYER:

            self.classifier = nn.Sequential(

                nn.Linear(
                    fusion_dim,
                    FUSION_HIDDEN_DIM
                ),

                nn.ReLU(
                    inplace=True
                ),

                nn.Linear(
                    FUSION_HIDDEN_DIM,
                    NUM_CLASSES
                ),

                #nn.Sigmoid()
            )

        else:

            self.classifier = nn.Sequential(

                nn.Linear(
                    fusion_dim,
                    NUM_CLASSES
                ),

                #nn.Sigmoid()
            )

    def forward(
        self,
        mel,
        frames,
        return_features=False
    ):

        audio_features, pa = (
            self.audio_branch(
                mel
            )
        )

        visual_features, pv = (
            self.visual_branch(
                frames
            )
        )

        fused = torch.cat(
            [
                audio_features,
                visual_features,
                pa,
                pv
            ],
            dim=1
        )

        output = self.classifier(
            fused
        )

        if return_features:

            return (
                output,
                audio_features,
                visual_features,
                pa,
                pv
            )

        return output

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    model = Model()

    mel = torch.randn(
        2,
        1,
        600,
        AUDIO_MFCC_COEFFICIENTS * 3
    )

    frames = torch.randn(
        2,
        15,
        3,
        224,
        224
    )

    output = model(
        mel,
        frames
    )

    print(
        "Output shape:",
        output.shape
    )

    output, audio_features, visual_features, pa, pv = model(
        mel,
        frames,
        return_features=True
    )

    print(
        "Audio features:",
        audio_features.shape
    )

    print(
        "Visual features:",
        visual_features.shape
    )

    print(
        "Audio confidence:",
        pa.shape
    )

    print(
        "Visual confidence:",
        pv.shape
    )