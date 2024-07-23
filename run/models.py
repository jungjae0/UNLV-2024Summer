import os

import torch
import pytorch_lightning as pl

from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import get_stats, iou_score, accuracy, precision, recall, f1_score

from configures import CFG

CLASSES = CFG['CLASSES']
ENCODER_WEIGHTS = CFG['ENCODER_WEIGHTS']


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()

        if model_name == 'UnetPlusPlus':
            model_UnetPlusPlus = UnetPlusPlus(
                        encoder_name='timm-regnety_120',
                        encoder_weights=ENCODER_WEIGHTS,
                        in_channels=3,
                        classes=len(CLASSES),
                        activation="softmax"
                    )

               
            self.model = model_UnetPlusPlus
        elif model_name == 'DeepLabV3Plus':
            
            model_DeepLabV3Plus = DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=len(CLASSES),
            activation="softmax"
        )
            
            self.model = model_DeepLabV3Plus
        self.lr = lr
        self.criterion = DiceLoss(mode="multiclass", from_logits=False)
        self.result = []
        self.model_name = model_name
        
        
    def forward(self, inputs, targets=None):
        if self.model_name == 'UnetPlusPlus':
            outputs = self.model(inputs)
            if targets is not None:
                loss = self.criterion(outputs, targets)
                tp, fp, fn, tn = get_stats(outputs.argmax(dim=1).unsqueeze(1).type(torch.int64), targets, mode='multiclass',
                                           num_classes=len(CLASSES))
                metrics = {
                    "Accuracy": accuracy(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "IoU": iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "Precision": precision(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "Recall": recall(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "F1score": f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
                }
                return loss, metrics, outputs
            else:
                return outputs
        elif self.model_name == 'DeepLabV3Plus':
            original_batch_size = inputs.size(0)

            if inputs.size(0) == 1:
                # 배치 크기가 1일 때 예외 처리
                inputs = inputs.repeat(2, 1, 1, 1)  # 같은 데이터를 두 번 반복하여 배치 크기를 2로 만듦
                outputs = self.model(inputs)
                outputs = outputs[:1]
                #outputs = outputs[0].unsqueeze(0)# 다시 배치 크기 1로 되돌림

                if targets is not None:# 차원을 맞춰서 배치 크기 2에 맞춰서 반복
                    if targets.dim() == 3:  # Assuming targets shape [C, H, W]
                        targets = targets.unsqueeze(0).repeat(2, 1, 1, 1)  # [B, C, H, W]
                    elif targets.dim() == 4:  # Assuming targets shape [B, C, H, W]
                        targets = targets.repeat(2, 1, 1, 1)  # [B*2, C, H, W]
                    loss = self.criterion(outputs.repeat(2,1,1,1),targets)
                    outputs = outputs[:1]  # outputs을 원래 배치 크기로 맞춤
                    targets = targets[:1]  # targets을 원래 배치 크기로 맞춤
            else:
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.criterion(outputs, targets)

            if targets is not None:
            # outputs = self.model(inputs)
            # if targets is not None:
            #     loss = self.criterion(outputs, targets)
                tp, fp, fn, tn = get_stats(outputs.argmax(dim=1).unsqueeze(1).type(torch.int64), targets, mode='multiclass',num_classes=len(CLASSES))
                metrics = {
                    "Accuracy": accuracy(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "IoU": iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "Precision": precision(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "Recall": recall(tp, fp, fn, tn, reduction="micro-imagewise"),
                    "F1score": f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
                }
                return loss, metrics, outputs

            else:
                return outputs

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=0.0001)
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


    def training_step(self, batch, batch_idx):
        images, masks = batch

        loss, metrics, outputs = self(images, masks)
        self.log_dict({
            "train/Loss": loss,
            "train/IoU": metrics['IoU'],
            "train/Accuracy": metrics['Accuracy'],
            "train/Precision": metrics['Precision'],
            "train/Recall": metrics['Recall'],
            "train/F1score": metrics['F1score']
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        loss, metrics, outputs = self(images, masks)
        self.log_dict({
            "val/Loss": loss,
            "val/IoU": metrics['IoU'],
            "val/Accuracy": metrics['Accuracy'],
            "val/Precision": metrics['Precision'],
            "val/Recall": metrics['Recall'],
            "val/F1score": metrics['F1score']
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch

        loss, metrics, outputs = self(images, masks)
        result = {
            "test/Loss": loss,
            "test/IoU": metrics['IoU'],
            "test/Accuracy": metrics['Accuracy'],
            "test/Precision": metrics['Precision'],
            "test/Recall": metrics['Recall'],
            "test/F1score": metrics['F1score']
        }
        result = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

        self.result.append(result)
        self.log_dict(result, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
