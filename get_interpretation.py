import pickle as pkl
import numpy as np
import random
import torch
import torchvision as tv

with open('./utils/heatmap_info/class_shap.pkl', "rb") as f:
    shap = pkl.load(f)

with open('./utils/heatmap_info/sc_idx.pkl', "rb") as f:
    sc_idx = pkl.load(f)

with open('./utils/heatmap_info/cos.pkl', "rb") as f:
    cos = pkl.load(f)

with open('./utils/heatmap_info/cos_gt.pkl', "rb") as f:
    cos_gt = pkl.load(f)

with open('./utils/www_img_val_80k_tem_adp_5_fc.pkl', "rb") as f:
    www_major_fc, _= pkl.load(f)

with open('./utils/www_img_val_80k_tem_adp_5_layer4.pkl', "rb") as f:
    www_major_l4, _ = pkl.load(f)

with open('./utils/www_img_val_80k_tem_adp_10_layer4_minor.pkl', "rb") as f:
    www_minor_l4, _ = pkl.load(f)

transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

examples_fc = tv.datasets.ImageFolder('./images/example_val_final', transform=transform)
examples_l4 = tv.datasets.ImageFolder('./images/example_val_layer4', transform=transform)
example_loader_fc = torch.utils.data.DataLoader(examples_fc, batch_size=1, shuffle=False)
example_loader_l4 = torch.utils.data.DataLoader(examples_l4, batch_size=1, shuffle=False)
interpret_idx = random.randint(0, len(sc_idx)-1)
print('Neurons to interpret: ', interpret_idx)

WWW_concepts = []
WWW_L4_major_concepts = []
WWW_L4_minor_concepts = []
GT = []
GT_idx = []
l4_idx=[]

all_words = []

with open('./utils/imagenet_labels.txt', 'r') as f:  # directory of imagenet_labels.txt
    words = (f.read()).split('\n')

for i in range(len(words)):
    temp=[]
    temp_words = words[i].split(', ')
    for word in temp_words:
        temp.append(f'{word}')
    all_words.append(temp)

correct = False

for i, major_idx in enumerate(interpret_idx):
    for gt_concept in all_words[major_idx]:
        if gt_concept in www_major_fc[major_idx]:
            correct = True
            break
    if correct:
        print(f'Neuron {i}: {major_idx}')
        print(f'Ground truth: {all_words[major_idx]}')
        print(f'WWW-fc major concept: {www_major_fc[major_idx]}')
        WWW_concepts.append(www_major_fc[major_idx])
        GT.append(all_words[major_idx])
        GT_idx.append(major_idx)
        
        WWW_temp_l4 = []
        WWW_temp_minor_l4 = []

        l4_important_idx = np.argsort(shap[major_idx], axis=0)[-3:]
        l4_idx.append(l4_important_idx)
        for j, minor_idx in enumerate(l4_important_idx):
            print(f'WWW-l4 major {j}: {www_major_l4[minor_idx]}')
            print(f'WWW-l4 minor {j}: {www_minor_l4[minor_idx]}')
            WWW_temp_l4.append(www_major_l4[minor_idx])
            WWW_temp_minor_l4.append(www_minor_l4[minor_idx])
        WWW_L4_major_concepts.append(WWW_temp_l4)
        WWW_L4_minor_concepts.append(WWW_temp_minor_l4)
    correct = False

concepts = []
concepts_l4 = []
for i in range(len(WWW_concepts)):
    concepts.append([GT_idx[i], WWW_concepts[i], GT[i]])
    concepts_l4.append([l4_idx[i], WWW_L4_major_concepts[i], WWW_L4_minor_concepts[i]])

with open('./utils/heat/imagenet/concept_FC.pkl', "wb") as f:
    pkl.dump(concepts, f)

with open('./utils/heat/imagenet/concept_L4.pkl', "wb") as f:
    pkl.dump(concepts_l4, f)
