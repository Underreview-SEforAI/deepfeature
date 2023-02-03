import torch
import torch.nn as nn
import numpy as np

class PGD():
    """
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=80/255,
                 alpha=20/255, steps=7, random_start=True):
        super(PGD, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self._targeted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        # loss = nn.functional.cosine_similarity()
        loss = nn.MSELoss()

        adv_images = images.clone().detach()

        _, clean_feature_map = self.model(images)
        cost_array = []
        adv_feature_map_array = []
        adv_images_array = []


        for idx in range(clean_feature_map.shape[1]):
            adv_images = images.clone().detach()
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs, adv_feature_map = self.model(adv_images)

                # Calculate loss
                # print(adv_feature_map[:,idx].shape)
                # cost = nn.functional.cosine_similarity(torch.flatten(adv_feature_map[:,idx],   1)
                #                                       ,torch.flatten(clean_feature_map[:,idx], 1), dim=0)
                # cost = cost.mean()
                cost = loss(adv_feature_map[:,idx], clean_feature_map[:,idx])
                # cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + self.alpha*grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            outputs, adv_feature_map = self.model(adv_images)
            adv_images_array.append(adv_images)
            adv_feature_map_array.append(adv_feature_map[:,idx])

        return adv_images_array, clean_feature_map, adv_feature_map_array

class RFGSM():
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 16/255)
        alpha (float): step size. (Default: 8/255)
        steps (int): number of steps. (Default: 1)
    """
    def __init__(self, model, eps=16/255, alpha=8/255, steps=1):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.random_start = True
        self.model = model
        self.steps = steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._targeted = False
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.MSELoss()
        adv_images = images.clone().detach()

        _, clean_feature_map = self.model(images)
        # print(clean_feature_map.shape)
        cost_array = []
        adv_feature_map_array = []
        adv_images_array = []

        for idx in range(clean_feature_map.shape[1]):
            adv_images = images.clone().detach()
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            
            adv_images.requires_grad = True
            outputs, adv_feature_map = self.model(adv_images)

            # Compute loss
            cost = loss(adv_feature_map[:,idx], clean_feature_map[:,idx])
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + (self.eps - self.alpha) * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            cost_array.append(cost.cpu().detach())
            outputs, adv_feature_map = self.model(adv_images)
            adv_images_array.append(adv_images)
            adv_feature_map_array.append(adv_feature_map[:,idx])
        # print(adv_images_array)
        return adv_images_array, _, _