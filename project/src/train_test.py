from configures import CFG
from dataloader import module_setup
from models import SegmentationModel

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

EPOCHS = CFG['EPOCHS']
IMAGE_SIZE = CFG['IMAGE_SIZE']
BATCH_SIZE = CFG['BATCH_SIZE']


def run(model_info):
    model_name = model_info['model_name']
    logger_name = model_info['logger_name']

    data_module = module_setup()

    model = SegmentationModel(model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val/F1score",
        mode="min"
    )

    logger = CSVLogger("lightning_logs", name=logger_name)

    early_stopping_callback = EarlyStopping(monitor="val/Accuracy", patience=5)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=31,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=EPOCHS,
        # accelerator="gpu",
        # devices=1
    )

    trainer.fit(model, data_module)




def main():
    models_info = {
        'UnetPlusPlus': {'model_name': 'UnetPlusPlus', 'logger_name': 'landcover-classification-log-unet'},
        'DeepLabV3Plus': {'model_name': 'DeepLabV3Plus', 'logger_name': 'landcover-classification-log-deeplab'},
        'SegNet': {'model_name': 'SegNet', 'logger_name': 'landcover-classification-log-segnet'}
    }

    model_info = models_info['SegNet']

if __name__ == '__main__':
    main()