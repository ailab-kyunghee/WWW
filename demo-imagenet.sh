
echo Model: ResNet-50 - ImageNet
echo D_probe: ImageNet-val
echo D_concept: WordNet nouns_80k
echo WWW
python ood_eval.py --name resnet50 --in-dataset imagenet --p-w 10 --p-a 10 --method taylor --clip_threshold 0.8

