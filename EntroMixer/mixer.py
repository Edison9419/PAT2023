import torch

Settings = [[2, 4, 4, 8], [1,2,4,8],[1,2,2,4]]


def shuffle(imgs: Tensor, score: float, setting):
    n = 0
    if score <= 0.1:
        n = setting[0]
    elif 0.1 < score <= 0.5:
        n = setting[1]
    elif 0.5 < score <= 0.9:
        n = setting[2]
    elif 0.9 < score <= 1:
        n = setting[3]
    slices = torch.chunk(imgs, n, 1)
    chunks = []
    for slice in slices:
        temp = torch.chunk(slice, n, dim=2)
        for t in temp:
            chunks.append(t)
    random.shuffle(chunks)
    res = []
    for i in range(n):
        res.append(torch.cat(chunks[n * i:n * (i + 1)], dim=1))
    return torch.cat(res, dim=2)


def rank(imgs, channle_w, spatial_W, k, vfe, thresholds):
    assert len(imgs.shape) == 4
    with torch.no_grad():
        slices = torch.chunk(imgs, k, 2)
        chunks = []
        num = int(imgs.shape[2] / k)
        channle_w = channle_w.repeat(1, 1, k, k)
        pooling = torch.nn.AvgPool2d(kernel_size=num, stride=num)
        spatial_W = pooling(spatial_W)
        scores = (channle_w * spatial_W).detach()
        scores = torch.mean(scores, dim=0).view(-1)
        for slice in slices:
            temp = torch.chunk(slice, k, dim=3)
            for t in temp:
                chunks.append(t)
        max = torch.max(scores)
        min = torch.min(scores)
        if vfe >= thresholds[1]:
            setting = Settings[0]
        elif vfe < thesholds[0]:
            setting = Settings[2]
        else:
            setting = Settings[1]
        for i in range(imgs.shape[1]):
            for j in range(len(chunks)):
                score = (scores[i * k * k + j] - min) / (max - min).item()
                chunks[j][:, i, :, :] = shuffle(chunks[j][:, i, :, :], score, setting)
        res = []
        for i in range(k):
            tmp = []
            for j in range(k):
                tmp.append(chunks[i * k + j])
            res.append(torch.cat(tmp, dim=3))
        return torch.cat(res, dim=2)
