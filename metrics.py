
import torch


def precision(target, prediction):
    """
    inputs are two boolean tensors of the same size, indicating pitches' presence.
    """

    result = torch.sum((target == prediction).masked_fill_(prediction == 0, 0)) / torch.sum(prediction)

    if torch.all(torch.isnan(result)):
        # torch.sum(prediction) == 0
        return 0

    return result


def recall(target, prediction):
    """
    inputs are two boolean tensors of the same size, indicating pitches' presence.
    """

    result = torch.sum((target == prediction).masked_fill_(target == 0, 0)) / torch.sum(target)

    if torch.all(torch.isnan(result)):
        # torch.sum(target) == 0
        return 1

    return result


if __name__ == "__main__":
    sample_target = torch.tensor([1, 1, 0, 0, 0, 0, 0])
    sample_prediction = torch.tensor([0, 1, 1, 1, 1, 0, 0])

    print(recall(sample_target, sample_prediction))