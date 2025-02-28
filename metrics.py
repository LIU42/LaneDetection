def iou(image1, image2, eps=1e-6):
    intersection = (image1 * image2).sum(dim=1)

    area1 = image1.sum(dim=1)
    area2 = image2.sum(dim=1)

    return intersection / (area1 + area2 - intersection + eps)


def dice(image1, image2, eps=1e-6):
    intersection = (image1 * image2).sum(dim=1)

    area1 = image1.sum(dim=1)
    area2 = image2.sum(dim=1)

    return (2 * intersection + eps) / (area1 + area2 + eps)


def precision(tp_count, fp_count):
    return tp_count / (tp_count + fp_count)


def recall(tp_count, fn_count):
    return tp_count / (tp_count + fn_count)


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)
