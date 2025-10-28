
%%capture
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, ViTFeatureExtractor, ViTModel
from PIL import Image
from tqdm import tqdm
import gc
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import VotingClassifier
import cv2
print(f"CV2 VERSION: {cv2.__version__}")
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import re
import random
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
import random
import os
from imblearn.under_sampling import RandomUnderSampler
from cuml.svm import SVR
import cuml
import cudf
import cupy as cp
import dask_cudf
import torch.nn as nn 
import joblib
from sklearn.metrics import mean_squared_error

def set_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)
    np.random.seed(seed)
    # Scikit-learn
    from sklearn.utils import check_random_state
    check_random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cp.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
from transformers import (ViTForImageClassification, ViTImageProcessor, DeiTForImageClassification, 
                          CvtForImageClassification, AutoFeatureExtractor, CLIPModel, CLIPProcessor, 
                          ViTMAEForPreTraining, AutoImageProcessor, ViTModel)
from accelerate import Accelerator
print('RAPIDS version',cuml.__version__)
torch.cuda.is_available()

df_train_metadata = pd.read_csv('isic-2024-challenge/train-metadata.csv')
accelerator = Accelerator()  # Automatically detects multi-GPU and mixed precision setups
device = accelerator.device  # Get the device from Accelerator

h5_file = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
base_dir = "/kaggle/input/2019-finetuned-vits/" 

vit_models = [
    ("google/vit-base-patch16-224", 64),
    ("facebook/deit-base-distilled-patch16-224", 64),
    ("microsoft/cvt-13", 64),
    ("facebook/dino-vitb16", 64),
    ("facebook/vit-mae-base", 32)
]

print("The device used:", device)


class ImageDataset(Dataset):
    def __init__(self, image_paths, feature_extractor):
        self.image_paths = image_paths
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Preprocess the image using the feature extractor
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        return inputs
    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is deleted
        self.h5_file.close()

def get_model_and_preprocessor(model_dir, num_labels=2, base_dir="/Users/jimmyhe/Desktop/KaggleCompetitions/ISISCANCER/HF_2019_finetuned/"):
    model_dir = model_dir.replace("/", "_").replace("-", "_")
    # Define the path to the model folder (no need to access individual files inside it)
    model_folder = os.path.join(base_dir, model_dir + '/')

    # Detect if there's a custom `best_model.pth` or use the default Hugging Face model
    model_path = os.path.join(model_folder, "best_model.pth")

    if "vit" in model_dir:
        model = ViTForImageClassification.from_pretrained(model_folder, num_labels=num_labels)
        preprocessor = ViTImageProcessor.from_pretrained(model_folder)
    elif "dino" in model_dir:
        model = ViTModel.from_pretrained(model_folder)
        preprocessor = ViTImageProcessor.from_pretrained(model_folder)
    elif "deit" in model_dir:
        model = DeiTForImageClassification.from_pretrained(model_folder, num_labels=num_labels)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_folder)
    elif "cvt" in model_dir:
        model = CvtForImageClassification.from_pretrained(model_folder, num_labels=num_labels)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_folder)
    elif "clip" in model_dir:
        model = CLIPModel.from_pretrained(model_folder)
        preprocessor = CLIPProcessor.from_pretrained(model_folder)
    elif "mae" in model_dir:
        model = ViTMAEForPreTraining.from_pretrained(model_folder)
        preprocessor = AutoImageProcessor.from_pretrained(model_folder)
    else:
        raise ValueError(f"Unsupported model: {model_dir}")

    # Check if custom weights exist (best_model.pth)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=accelerator.device))
    
    model.eval()  # Set the model to evaluation mode

    return model, preprocessor

def get_image_embeddings(model_name='', batch_size=32, image_paths=None):
    all_embeddings = []


    model, feature_extractor = get_model_and_preprocessor(model_name)
    dataset = ImageDataset(image_paths, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.to(accelerator.device)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            pixel_values = batch["pixel_values"].to(accelerator)
            with torch.cuda.amp('cuda',enabled=True):
                model_output = model(pixel_values=pixel_values)
            embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
            if 'cvt' in model_name:
                avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
                embeddings = avg_pool(model_output.last_hidden_state).squeeze()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy())

        print(f"{model_name} has embedding shape:", embeddings.shape)

    del dataset, dataloader, model
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(all_embeddings)

def sanitize_filename(name):
    # Replace '/' and '-' with '_'
    name = name.replace('/', '_').replace('-', '_')
    # Remove any other non-alphanumeric characters (except underscore)
    return re.sub(r'[^\w\-_\.]', '', name)

image_paths = [f"/content/train-image/image/{id}.jpg" for id in df_train_metadata.isic_id]
image_paths = image_paths[:10]
all_train_embeds = []
for model_name, batch_size in vit_models:
    all_embeddings = []
    embeddings = get_image_embeddings(model_name=model_name, batch_size=batch_size, image_paths=image_paths)
    all_embeddings.append(embeddings)
    all_embeddings = np.vstack(all_embeddings)
    save_directory = "/Users/jimmyhe/Desktop/KaggleCompetitions/ISISCANCER/HF_2019_finetuned_embeddings/"
    os.makedirs(save_directory, exist_ok=True)

    # Sanitize the model_name to ensure it's a valid filename
    safe_model_name = sanitize_filename(model_name)
    file_path = os.path.join(save_directory, f"finetuned_{safe_model_name}_image_embeddings.npy")
    all_train_embeds.append(all_embeddings)
    print(f"Attempting to save to: {file_path}")
    print(f"embedding shape for {model_name}:",all_embeddings.shape)
    try:
        np.save(file_path, all_embeddings)
        print(f"Successfully saved embeddings for {model_name}")
    except Exception as e:
        print(f"Error saving embeddings for {model_name}: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(save_directory)}")



def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


X = np.concatenate(all_train_embeds, axis=1)
y = df_train_metadata['target'].values

FOLDS = 5
skf_svr = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

svr_oof = np.zeros(len(X), dtype='float32')
svr_fold_scores = []
mse_fold_scores = []
trained_models = []
for fold, (train_index, val_index) in enumerate(skf_svr.split(X, y)):
    print('#'*50)
    print(f'### Fold {fold+1}')
    print('#'*50)

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    print(f"Before undersampling:")
    print(f"Training set shape: {X_train.shape}, Training set distribution: {np.bincount(y_train)}")
    print(f"Validation set shape: {X_val.shape}, Validation set distribution: {np.bincount(y_val)}")

    # Undersample only the training data
    undersampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    print(f"\nAfter undersampling:")
    print(f"Training set shape: {X_train_resampled.shape}, Training set distribution: {np.bincount(y_train_resampled)}")
    # indices = undersampler.sample_indices_
    # positive_train_indices = np.where(y_train_resampled == 1)[0]
    # for i, index in enumerate(positive_train_indices):
    #   start_index = i * 5
    #   end_index =((i + 1) * 5) -2
    #   new_aug_embed = augmented_pos_img_embeddings[start_index:end_index]
    #   X_train_resampled = np.vstack((X_train_resampled, new_aug_embed))
    #   y_train_resampled = np.append(y_train_resampled, [1]*new_aug_embed.shape[0])
    # print(f"\nAfter augmentation:")
    # print(f"Training set shape: {X_train_resampled.shape}, Training set distribution: {np.bincount(y_train_resampled)}")

    print(f"Validation set shape: {X_val.shape}, Validation set distribution: {np.bincount(y_val)}")


    X_train_cp = cp.asarray(X_train_resampled)
    X_val_cp = cp.asarray(X_val)
    y_train_resampled = cp.asarray(y_train_resampled)

    # Initialize and train RAPIDS SVR
    model = SVR(
        C=1.0,
        epsilon=0.1,
        kernel='rbf',
        cache_size=4096,
        max_iter=1000,
        tol=1e-3,
        verbose=True
    )

    model.fit(X_train_cp, y_train_resampled)

    preds = model.predict(X_val_cp)
    print(preds[:10])
    preds = (preds - preds.min()) / (preds.max() - preds.min())

    # Move predictions back to CPU for scoring
    preds_cpu = cp.asnumpy(preds)
    svr_oof[val_index] = preds_cpu
    fold_score = comp_score(pd.DataFrame(y_val), pd.DataFrame(preds_cpu))
    svr_fold_scores.append(fold_score)
    mse_score = mean_squared_error(y_val, preds_cpu)
    mse_fold_scores.append(mse_score)
    trained_models.append(model)

    print(f"Fold {fold+1} MSE: {mse_score}")

    print(f"\n=> Fold score: {fold_score}")
    print("\n")

print(f"Average MSE: {np.mean(mse_fold_scores)}")
print(f"Overall MSE: {mean_squared_error(y, svr_oof)}")
print('#'*50)
overall_score = comp_score(pd.DataFrame(y), pd.DataFrame(svr_oof))
print(f'Mean fold score = {np.mean(svr_fold_scores)}')
print(f'Overall CV score = {overall_score}')

# Specify the folder where the models will be saved
output_folder = '/Users/jimmyhe/Desktop/KaggleCompetitions/ISISCANCER/VisualEmbeddingFinal.ipynb/final_svr/'

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save each model into the specified folder
for i, model in enumerate(trained_models):
    model_path = os.path.join(output_folder, f'svr_model_fold_{i+1}.joblib')
    joblib.dump(model, model_path)