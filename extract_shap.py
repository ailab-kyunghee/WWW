import argparse
import torchvision as tv
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='./datasets/ILSVRC-2012/train', help='Path to ImageNet data')
    parser.add_argument('--shap_save_root', default='./utils/class_shap.pkl', help='Path to Shapley value matrix data')

    return parser.parse_args()


def main():
    
    args = parse_args()

    ## Load model ##
    ##### ResNET50 #####
    from models.resnet import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = model.cuda()
    model.eval()
    featdim = 2048
    class_dim = 1000

    transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    traindata = tv.datasets.ImageFolder(args.data_root, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    if not os.path.exists(args.shap_save_root):
        class_num = 0
        shap = []
        shap_class = np.zeros((class_dim, featdim)) 
        shap_temp = np.zeros(featdim)
        for image, labels in tqdm(trainloader):
            image = image.cuda()
            
            if class_num != labels:
                for i in range(len(shap)):
                    shap_temp += shap[i].squeeze()    
                shap_class[class_num,:] = shap_temp / len(shap)
                shap = []
                shap_temp = np.zeros(featdim)
                class_num += 1
                    
            shap_batch  = model._compute_taylor_scores(image, labels)
            shap.append(shap_batch[0][0].squeeze().cpu().detach().numpy())

        for i in range(len(shap)):
            shap_temp += shap[i].squeeze()    
        shap_class[class_num,:] = shap_temp / len(shap)
        shap = []

        with open(args.shap_save_root, 'wb') as f:
            pickle.dump(shap_class, f)

if __name__ == '__main__':
    main()