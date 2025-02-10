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
