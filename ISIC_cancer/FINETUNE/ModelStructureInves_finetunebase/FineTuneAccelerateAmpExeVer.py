from transformers import (
    ViTForImageClassification,
    ViTModel,
    ViTImageProcessor,
    DeiTForImageClassification,
    ViTImageProcessor,
    CLIPModel,
    CLIPProcessor,
    AutoFeatureExtractor,
    CvtForImageClassification,
    AutoImageProcessor,
    ViTMAEForPreTraining,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             auc,
                             accuracy_score,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import pandas as pd
import os
from tqdm import tqdm
import numpy as np 
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import torch.nn.functional as F

import torch
from accelerate import Accelerator
import timm 

# from libauc.losses import pAUC_CVaR_Loss
# from libauc.optimizers import SOPA
# from libauc.sampler import DualSampler  # data resampling (for binary class)
# from libauc.metrics import pauc_roc_score

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A
import torch.nn as nn 
import logging 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this function at the start of your main function
set_seed()


num_epochs = 25


def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

        
        



def get_model_and_preprocessor(model_name, num_labels):
    if "vit" in model_name:
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = ViTImageProcessor.from_pretrained(model_name)
    elif "dino" in model_name:
        model = ViTModel.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = ViTImageProcessor.from_pretrained(model_name)
    elif "deit" in model_name:
        model = DeiTForImageClassification.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
    elif "cvt" in model_name:
        model = CvtForImageClassification.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
    elif "timm" in model_name:
        model = timm.create_model(model_name.split('/')[-1], pretrained=True, num_classes=num_labels)
        data_config = timm.data.resolve_model_data_config(model)
        preprocessor = timm.data.create_transform(**data_config)  
    elif "clip" in model_name:
        model = CLIPModel.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = CLIPProcessor.from_pretrained(model_name)
    elif "mae" in model_name:
        model = ViTMAEForPreTraining.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        preprocessor = AutoImageProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model, preprocessor




log_file = 'training_log.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Compute pAUC score
def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, feature_extractor, mode='train'):
        self.data = csv_file if isinstance(csv_file, pd.DataFrame) else pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_paths = self.data['image'].values
        self.labels = self.data['target'].values
        self.feature_extractor = feature_extractor
        if mode == 'train':
            self.transform = A.Compose([
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, p=0.75),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.0),
                    A.ElasticTransform(alpha=3),
                ], p=0.7),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.CoarseDropout(max_holes=1, max_height=int(224 * 0.375), max_width=int(224 * 0.375), p=0.7),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)*5
    
    def __getitem__(self, idx):
        idx = idx//5
        img_path = os.path.join(self.img_dir, self.image_paths[idx])
        if not os.path.exists(img_path):
            img_path += '.jpg'
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['pixel_values'] = F.interpolate(inputs['pixel_values'], size=(224, 224), mode='bilinear', align_corners=False)

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        inputs['pixel_values'] = inputs['pixel_values'].bfloat16()
        return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

def train_model(model, model_name, train_dataloader, val_dataloader, optimizer, scheduler, accelerator,save_dir, num_epochs=3):
    model.train()
    best_pauc = 0
    loss_log = []
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_losses = {'epoch': epoch + 1, 'train_loss': 0, 'train_pauc': 0, 'val_loss': 0, 'val_pauc': 0}
        
        # Training loop
        model.train()
        train_preds, train_labels = [], []
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch"):
            with accelerator.accumulate(model):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}  # Move batch to device
                outputs = model(**batch)
                loss = loss_fn(outputs.logits[:, 1], batch['labels'].float())
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_losses['train_loss'] += loss.item()
                train_preds.extend(torch.sigmoid(outputs.logits[:, 1]).detach().cpu().numpy())
                train_labels.extend(batch['labels'].cpu().numpy())

        epoch_losses['train_loss'] /= len(train_dataloader)
        epoch_losses['train_pauc'] = comp_score(pd.Series(train_labels), pd.Series(train_preds))
        epoch_losses['train_accuracy'] = accuracy_score(train_labels, np.round(train_preds))
        epoch_losses['train_MSE'] = mean_squared_error(train_labels, train_preds)
        # epoch_losses['train_fpr'], epoch_losses['train_tpr'], _ = roc_curve(train_labels, train_preds)
        # epoch_losses['train_auc'] = auc(epoch_losses['train_fpr'], epoch_losses['train_tpr'])


  
        # Validation loop
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", unit="batch"):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}  # Move batch to device
                outputs = model(**batch)
                loss = loss_fn(outputs.logits[:, 1], batch['labels'].float())
                
                epoch_losses['val_loss'] += loss.item()
                val_preds.extend(torch.sigmoid(outputs.logits[:, 1]).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())

        epoch_losses['val_loss'] /= len(val_dataloader)
        epoch_losses['val_pauc'] = comp_score(pd.Series(val_labels), pd.Series(val_preds))
        epoch_losses['val_accuracy'] = accuracy_score(val_labels, np.round(val_preds))
        epoch_losses['val_MSE'] = mean_squared_error(val_labels, val_preds)
        # epoch_losses['val_fpr'], epoch_losses['val_tpr'], _ = roc_curve(val_labels, val_preds)
        # epoch_losses['val_auc'] = auc(epoch_losses['val_fpr'], val_labels['val_tpr'])

        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}, "
                          f"Train Loss: {epoch_losses['train_loss']:.4f}, "
                          f"Train pAUC: {epoch_losses['train_pauc']:.4f}, "
                          f"Train Accuracy: {epoch_losses['train_accuracy']:.4f}, "
                          f"Train MSE: {epoch_losses['train_MSE']:.4f}, "
                        #   f"Train fpr: {epoch_losses['train_fpr']:.4f}, "
                        #   f"Train tpr: {epoch_losses['train_tpr']:.4f}, "
                        #   f"Train AUC: {epoch_losses['train_auc']:.4f}"
                          f"Val Loss: {epoch_losses['val_loss']:.4f}, "
                          f"Val pAUC: {epoch_losses['val_pauc']:.4f}"
                          f"Val Accuracy: {epoch_losses['val_accuracy']:.4f}, "
                          f"Val MSE: {epoch_losses['val_MSE']:.4f}, "
                        #   f"Val fpr: {epoch_losses['val_fpr']:.4f}, "
                        #   f"Val tpr: {epoch_losses['val_tpr']:.4f}, "
                        #   f"Val AUC: {epoch_losses['val_auc']:.4f}"
        )
        

        # Save best model
        if epoch_losses['val_pauc'] > best_pauc:
            best_pauc = epoch_losses['val_pauc']
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f"{save_dir}/best_model_{model_name.replace('/', '_')}.pth")

        # Save model at each epoch
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), f"{save_dir}/model_epoch_{epoch + 1}_{model_name.replace('/', '_')}.pth")

        # Log losses to CSV
        loss_log.append(epoch_losses)
        if accelerator.is_main_process:
            pd.DataFrame(loss_log).to_csv(f"{save_dir}/loss_log_{model_name.replace('/', '_')}.csv", index=False)

    accelerator.print(f"Best Validation pAUC Score: {best_pauc:.4f}")
    return best_pauc

def main():
    accelerator = Accelerator()
    DEVICE = accelerator.device
    logger.info(f"DEVICE USING: {DEVICE}")

    train_img = './2019_Train/image/'
    train_metadata_19 = pd.read_csv('./ISIC_2019_Training_GroundTruth.csv')
    train_metadata_19['target'] = train_metadata_19[['BCC', 'SCC', 'MEL']].eq(1.0).any(axis=1).astype(int)
    # train_metadata_19 = train_metadata_19.head(200)  # Adjust as needed

    vit_models = [
        ("google/vit-base-patch16-224",64),
        ("facebook/deit-base-distilled-patch16-224", 64),
        ("microsoft/cvt-13", 64),
        ("facebook/dino-vitb16", 64),
        #("openai/clip-vit-base-patch32", 16),
        # ("timm/vit_small_patch16_224", 64),
        ("facebook/vit-mae-base", 32)
    ]

    for model_name, batch_size in vit_models:
        model, feature_extractor = get_model_and_preprocessor(model_name, num_labels=2)
        logger.info('#'*25)
        logger.info(f"Processing model: {model_name}")
        logger.info('#'*25)


        # if 'timm' in model_name.lower():
        #     model = timm.create_model(model_name, pretrained=True, num_classes=2)
        #     feature_extractor = timm.data.create_transform(
        #         input_size=224,
        #         is_training=True,
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225)
        #     )
        # else:
            
        #     feature_extractor = ViTImageProcessor.from_pretrained(model_name,force_download=True)
        #     model = ViTForImageClassification.from_pretrained(
        #         model_name,
        #         num_labels=2,
        #         ignore_mismatched_sizes=True
        #     )

        train_df, val_df = train_test_split(train_metadata_19, test_size=0.2, random_state=42)

        train_dataset = ImageDataset(csv_file=train_df, img_dir=train_img, feature_extractor=feature_extractor, mode='train')
        val_dataset = ImageDataset(csv_file=val_df, img_dir=train_img, feature_extractor=feature_extractor, mode='val')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        num_training_steps = len(train_dataloader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        save_dir = f"./models/finetuned_model_{model_name.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )

        train_model(model, model_name, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, save_dir,num_epochs=num_epochs)
        # Save the finetuned model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        if accelerator.is_main_process:
            feature_extractor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
