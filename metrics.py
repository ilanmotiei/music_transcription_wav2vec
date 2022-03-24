
import torch


def precision(target, prediction):
    """
    inputs are two boolean tensors of the same size, indicating pitches' presence.
    """

    try:
        return torch.sum((target == prediction).masked_fill_(prediction == 0, 0)) / torch.sum(prediction)
    except ZeroDivisionError:
        return 1


def recall(target, prediction):
    """
    inputs are two boolean tensors of the same size, indicating pitches' presence.
    """

    try:
        return torch.sum((target == prediction).masked_fill_(target == 0, 0)) / torch.sum(target)
    except ZeroDivisionError:
        return 1


if __name__ == "__main__":
    sample_target = torch.tensor([1, 1, 0, 0, 0, 0, 0])
    sample_prediction = torch.tensor([0, 1, 1, 1, 1, 0, 0])

    print(recall(sample_target, sample_prediction))