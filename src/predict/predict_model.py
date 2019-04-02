from PIL import Image
import torch
from torch import nn
from torchvision import transforms


def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224, 224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out


def predict_dog_prob_of_single_instance(model, tensor):
    batch = torch.stack([tensor])
    softMax = nn.Softmax(dim=1)
    preds = softMax(model(batch))
    return preds[0, 1].item()


def test_data_from_fname(fname):
    # im = Image.open('{}/{}'.format(test_data_dir, fname))
    im = Image.open(fname)
    return apply_test_transforms(im)
