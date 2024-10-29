import torch
from torchvision.models import vgg19



class Vgg19(torch.nn.Module):
    def __init__(self, vgg19_weights):
        super(Vgg19, self).__init__()
        vgg = vgg19(pretrained=False)
        params = torch.load(vgg19_weights)
        vgg.load_state_dict(params)
        features = list(vgg.features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)

            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))



def build_perceptual_loss(x):
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim=0)
    l = l.sum()
    return l


def build_gram_matrix(x):
    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram


def build_style_loss(x):
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] * f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim=0)
    l = l.sum()
    return l


def build_vgg_loss(x):
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style
