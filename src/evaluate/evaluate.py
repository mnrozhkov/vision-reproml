from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch.autograd import Variable
import tqdm


def predict_image(model, image: torch.Tensor, device) -> int:

    image_tensor = image.float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()

    return index


def evaluate(model, dataloader, device):

    y_pred = []
    y_true = []
    # read by batches
    for images, labels in tqdm.tqdm(dataloader):
        # for each image in batch_size
        for ii in range(len(images)):
            image = images[ii]
            # predict image
            index = predict_image(model, image, device)
            # append y_pred and y_true
            y_pred.append(index)
            y_true.append(int(labels[ii]))

    return ({
        'precision': precision_score(y_pred=y_pred, y_true=y_true),
        'recall': recall_score(y_pred=y_pred, y_true=y_true),
        'f1': f1_score(y_pred=y_pred, y_true=y_true),
        'roc_auc': roc_auc_score(y_score=y_pred, y_true=y_true),
        'cm': confusion_matrix(y_pred=y_pred, y_true=y_true)
    })
