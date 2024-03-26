import argparse
import torchvision as tv
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='./datasets/ILSVRC-2012/val', help='Path to ImageNet data')
    parser.add_argument('--model', default='rn50', help='model name')
    parser.add_argument('--layer', default='fc', help='target layer')
    parser.add_argument('--save_root', default='./utils', help='Path to idx')
    parser.add_argument('--img_save_root', default='./images/example_val_final', help='Path to saved img')
    parser.add_argument('--num_example', default=40, type=int, help='# of examples to be used')
    parser.add_argument('--num_act', default=1, type=int, help='# of examples to be used')

    return parser.parse_args()


def main():
    
    args = parse_args()
    layer = args.layer
    args.image_save_root = f'./images/example_val_{layer}'

    ## Load model ##
    ##### ResNET50 #####
    if args.model == 'rn50':
        from models.resnet import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)

    ##### ViT-B 16  #####
    elif args.model == 'vitb16':
        from models.ViT import _create_vision_transformer
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **dict(model_kwargs))

    ##### ResNet-18  #####
    elif args.model == 'rn18':
        model = tv.models.resnet18(num_classes=365)
        state_dict = torch.load(f'{args.save_root}\\resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        model.load_state_dict(new_state_dict)


    model.cuda().eval()

    # if not os.path.exists(f"{args.save_root}\\slice_act_{args.num_act-1}"):
    if layer == 'fc':
        transform = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
            ])

        traindata = tv.datasets.ImageFolder(args.data_root, transform=transform)
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
        with torch.no_grad():
            act_matrix = []
            counter = 0

            for batch_idx, (image, labels) in enumerate(tqdm(trainloader)):
                image = image.cuda()
                act_matrix.append(model(image).squeeze().cpu().detach().numpy())
                if batch_idx % int(len(trainloader)/args.num_act) == 0 and batch_idx != 0:
                    act_matrix = np.concatenate(act_matrix, axis=0)
                    with open(f"{args.save_root}\\slice_act_{counter}_{layer}", 'wb') as f:
                        pickle.dump(act_matrix, f)
                    counter += 1
                    act_matrix = []
            if args.num_act == 1:
                act_matrix = np.concatenate(act_matrix, axis=0)
                with open(f"{args.save_root}\\slice_act_{counter}_{layer}", 'wb') as f:
                    pickle.dump(act_matrix, f)
    elif layer == 'l4':
        transform = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
            ])

        traindata = tv.datasets.ImageFolder(args.data_root, transform=transform)
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
        with torch.no_grad():
            act_matrix = []
            counter = 0

            for batch_idx, (image, labels) in enumerate(tqdm(trainloader)):
                image = image.cuda()
                act_matrix.append(model.extract_feature_4(image).squeeze().cpu().detach().numpy())
                if batch_idx % int(len(trainloader)/args.num_act) == 0 and batch_idx != 0:
                    act_matrix = np.concatenate(act_matrix, axis=0)
                    with open(f"{args.save_root}\\slice_act_{counter}_{layer}", 'wb') as f:
                        pickle.dump(act_matrix, f)
                    counter += 1
                    act_matrix = []
            if args.num_act == 1:
                act_matrix = np.concatenate(act_matrix, axis=0)
                with open(f"{args.save_root}\\slice_act_{counter}_{layer}", 'wb') as f:
                    pickle.dump(act_matrix, f)

    if True:
        offset = 0
        for i in tqdm(range(args.num_act)):
            with open(f"{args.save_root}\\slice_act_{i}_{layer}", 'rb') as f:
                act_matrix = pickle.load(f)
            
            idx_matrix = np.zeros((args.num_example, act_matrix.shape[1]))
            feat_matrix = np.zeros((args.num_example, act_matrix.shape[1]))

            for j in range(act_matrix.shape[1]):
                sorted_idx = np.argsort(act_matrix[:,j])
                top_idx = np.flip(sorted_idx[-args.num_example:])
                idx_matrix[:,j] = top_idx + offset
                feat_matrix[:,j] = act_matrix[top_idx,j]

            with open(f"{args.save_root}\\idx_{args.num_example}_{i}_{layer}.pkl", 'wb') as f:
                pickle.dump(idx_matrix, f)

            with open(f"{args.save_root}\\feat_{args.num_example}_{i}_{layer}.pkl", 'wb') as f:
                pickle.dump(feat_matrix, f)
            offset += len(act_matrix)

    if True:
        idx_mats = []
        feat_mats = []
        idx_matrix = np.zeros((args.num_example, act_matrix.shape[1]))

        for i in range(args.num_act):
            with open(f"{args.save_root}\\idx_{args.num_example}_{i}_{layer}.pkl", 'rb') as f:
                idx_matrix = pickle.load(f)
            with open(f"{args.save_root}\\feat_{args.num_example}_{i}_{layer}.pkl", 'rb') as f:
                feat_matrix = pickle.load(f)
            idx_mats.append(idx_matrix)
            feat_mats.append(feat_matrix)

        idx_mats = np.concatenate(idx_mats, axis=0)
        feat_mats = np.concatenate(feat_mats, axis=0)

        for j in range(idx_matrix.shape[1]):
            sorted_idx = np.argsort(feat_mats[:,j])
            top_idx = np.flip(sorted_idx[-args.num_example:])
            idx_matrix[:,j] = idx_mats[top_idx,j]
        
        with open(f"{args.save_root}\\slice_idx_{args.num_example}_{layer}.pkl", 'wb') as f:
            pickle.dump(idx_matrix, f)

    if True:
        traindata = tv.datasets.ImageFolder(args.data_root)
        for i in range(idx_matrix.shape[1]):
            os.makedirs(f'{args.img_save_root}/{i:04d}', exist_ok=True)
            if i % 100 == 0:
                print(i)
            for j in range(args.num_example):
                image, labels = traindata[int(idx_matrix[j,i])]
                img_dir = f'{args.img_save_root}/{i:04d}/{i:04d}_{j:02d}_{labels:03d}.jpg'
                image.save(img_dir)


if __name__ == '__main__':
    main()




