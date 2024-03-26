import torch
import numpy as np
import cv2
import torchvision as tv

def load_images(img_dir):
    imgset = tv.datasets.ImageFolder(img_dir)
    return imgset

def compute_nams(model, images, feature_index):
    b_size = images.shape[0]
    feature_maps = model.extract_feature_map(images)
    nams = (feature_maps[:, feature_index, :, :]).detach()
    nams_flat = nams.view(b_size, -1) 
    nams_max, _ = torch.max(nams_flat, dim=1, keepdim=True)
    nams_flat = nams_flat/nams_max
    nams = nams_flat.view_as(nams)

    nams_resized = []
    for nam in nams:
        nam = nam.cpu().numpy()
        nam = cv2.resize(nam, images.shape[2:])
        nams_resized.append(nam)
    nams = np.stack(nams_resized, axis=0)
    nams = torch.from_numpy(1-nams)
    return nams

def compute_heatmaps(imgs, masks):
    heatmaps = []
    for (img, mask) in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = torch.from_numpy(heatmaps).permute(0, 3, 1, 2)
    return heatmaps

def create_images(img_dir, feature_index, model):
    images_high_set = load_images(img_dir)
    image_loader = torch.utils.data.DataLoader(images_high_set, batch_size=1, shuffle=False)

    for images, _ in image_loader:
        images_nams = compute_nams(model, images, feature_index)
        images_heatmaps = compute_heatmaps(images, images_nams)
        

    return images_highest, images_heatmaps
