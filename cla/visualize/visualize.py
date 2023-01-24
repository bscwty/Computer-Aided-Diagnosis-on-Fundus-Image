import torch
from torch.nn import ReLU
import torch.nn.functional as F

import copy
import os
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

class GradCAM:

    def __init__(self, arch: torch.nn.Module):

        self.model_arch = arch
        self.model_arch.eval()

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.model_arch.backbone.features.register_forward_hook(forward_hook)
        self.model_arch.backbone.features.register_backward_hook(backward_hook)

    def saliency_map_size(self, *input_size):

        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, threshold, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()
        logit = self.model_arch(input)
        logit = torch.sigmoid(logit)

        pNum = int(torch.sum(logit > threshold))
        scores, idx = torch.sort(logit, dim=-1, descending=True)
        scores, idx = scores.squeeze(), idx.squeeze()
        scores, idx = scores[:5], idx[:5]

        saliency_maps = []
        for i, score in enumerate(scores):

            retain_graph = True if i < len(scores) - 1 else False

            self.model_arch.zero_grad()
            score.backward(retain_graph=retain_graph)
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()

            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

            saliency_maps.append(saliency_map)

        return pNum, saliency_maps, scores, idx

    def __call__(self, input, threshold=0.5, class_idx=None, retain_graph=False):
        return self.forward(input, threshold, class_idx, retain_graph)


class GradCAMpp(GradCAM):

    def __init__(self, arch):
        super(GradCAMpp, self).__init__(arch)

    def forward(self, input, threshold, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()
        logit = self.model_arch(input)
        logit = torch.sigmoid(logit)

        pNum = int(torch.sum(logit > threshold))
        scores, idx = torch.sort(logit, dim=-1, descending=True)
        scores, idx = scores.squeeze(), idx.squeeze()
        scores, idx = scores[:5], idx[:5]

        saliency_maps = []

        for i, score in enumerate(scores):

            retain_graph = True if i < len(scores) - 1 else False

            self.model_arch.zero_grad()
            score.backward(retain_graph=retain_graph)
            gradients = self.gradients['value']  # dS/dA
            activations = self.activations['value']  # A
            b, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                    activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom+1e-7)
            positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.upsample(saliency_map, size=(448, 448), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

            saliency_maps.append(saliency_map)

        return pNum, saliency_maps, scores, idx

class CAM(GradCAM):

    def __init__(self, arch):
        super(CAM, self).__init__(arch)

    def forward(self, input, threshold, class_idx=None, retain_graph=False):

        with torch.no_grad():

            b, c, h, w = input.size()
            logit = self.model_arch(input)
            logit = torch.sigmoid(logit)

            pNum = int(torch.sum(logit > threshold))
            scores, idx = torch.sort(logit, dim=-1, descending=True)
            scores, idx = scores.squeeze(), idx.squeeze()
            scores, idx = scores[:5], idx[:5]

            activations = self.activations['value'].squeeze().reshape(-1, 196)
            weights = list(self.model_arch.parameters())[-1]
            weights = weights[idx]
            cam = torch.matmul(weights, activations).reshape(-1, 14, 14)#weights * activations

            saliency_maps = []

            for saliency_map in cam:

                saliency_map = F.relu(saliency_map).unsqueeze(0).unsqueeze(0)
                saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                saliency_maps.append(saliency_map)

        return pNum, saliency_maps, scores, idx

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = self.model.backbone.features.conv0#list(self.model.backbone.features.named_modules())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)

            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        #print(self.model.backbone)
        for pos, module in self.model.backbone.features.named_modules():
            #print(module)
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image: torch.Tensor, target_class, device):
        # Forward pass
        input_image.requires_grad = True
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('./cam_imgs', file_name + '.png')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def apply_heatmap(R, sx, sy):
    """
        Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial
        This is (so far) only used for LRP
    """
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heatmap
    # plt.show()


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)
