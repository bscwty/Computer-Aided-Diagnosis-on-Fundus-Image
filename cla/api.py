import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T

import random
from PIL import Image
import numpy as np

from cla.model import Model
from cla.configs.config import getConfig
from cla.visualize.visualize import GradCAM, GradCAMpp, CAM, GuidedBackprop, save_gradient_images
from cla.visualize.utils import visualize_cam, guided_normalize


import matplotlib.pyplot as plt

def get_transforms():

    normalize = T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))

    transforms = T.Compose([
        T.Resize(448),
        T.ToTensor(),
        normalize])

    display_transforms = T.Compose([
        T.Resize(448),
        T.ToTensor()])

    return transforms, display_transforms


class Cla:

    def __init__(self, model_path) -> None:

        self.args = getConfig('./cla/configs/kaggle.yaml')
        self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_path = model_path
        self.diseases = ['DR          ', 
                         'Glaucoma    ',
                         'AMD         ',
                         'Hypertension',
                         'Myopia      ',
                         'Macula      ',
                         'Disc        ',
                         'Others      ']

    def init_net(self):

        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

        arch = Model(8, 'densenet').to(self.args.device)
        # arch.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        '''
        TODO
        '''
        arch.load_state_dict(torch.load(self.model_path))
        arch = arch.to(self.args.device)
        arch.eval()

        self.arch = arch

    def _load_tool(self, method):

        if method == 'gradcam':
            tool = GradCAM(self.arch)
        elif method == 'gradcampp':
            tool = GradCAMpp(self.arch)
        elif method == 'cam':
            tool = CAM(self.arch)
        elif method == 'guided-gradcam':
            tool = GradCAM(self.arch)

        return tool

    def _get_guidedmask(self, im, idx, GBP):
        guided_mask = []
        for i in idx:
            g_mask = GBP.generate_gradients(im, i, self.args.device)
            guided_mask.append(g_mask)

        return guided_mask

    def list_content(self, pNum, scores, idx):

        content = []
        if pNum:
            for p, (s, i) in enumerate(zip(scores, idx)):
                if p == pNum:
                    content.append('=====================')
                if float(s) < 0.1:
                    content.append('%s    %.2f%s'%(self.diseases[i], float(s)*100, '%'))
                else:    
                    content.append('%s   %.2f%s'%(self.diseases[i], float(s)*100, '%'))

        else:
            content.append('Normal')
            content.append('=====================')
            for s, i in zip(scores, idx):
                content.append(str(self.diseases[i]))
                # if float(s) < 0.1:
                #     content.append('%s    %.2f%s'%(self.diseases[i], float(s)*100, '%'))
                # else:    
                #     content.append('%s   %.2f%s'%(self.diseases[i], float(s)*100, '%'))

        return content

    def forward(self, img: Image.Image, method):

        tool = self._load_tool(method)
        transforms, display_transforms = get_transforms()

        im = transforms(img).to(self.args.device)
        img = display_transforms(img).to(self.args.device)

        im = im.unsqueeze(0)
        pNum, mask, scores, idx = tool(im)

        imgs = []
        disease = []

        if not pNum: # Normal
            mask.insert(0, torch.zeros_like(mask[0]))

        heatmaps, results = visualize_cam(mask, img)

        for result in results:

            img = T.ToPILImage()(result)
            imgs.append(img)

        # if not pNum:
        #     imgs = [Image.fromarray(np.zeros(shape=(512, 512)))] + imgs[:4]

        content = self.list_content(pNum, scores, idx)

        return pNum, imgs, content

    def __call__(self, img: Image.Image, method='cam'):
        return self.forward(img, method)


def main():

    #display(args, display_method)

    img_path = '/mnt/data1/MedicalDataset/Kaggle/valid/10899_right.jpeg'

    folder = './example_imgs'

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        org_img = Image.open(img_path).convert('RGB')

        display = Cla()
        imgs_cam, idx1 = display(img_path, 'cam')
        imgs_gradcam, idx2 = display(img_path, 'gradcam')
        imgs_gradcampp, idx3 = display(img_path, 'gradcampp')

        identified_disease_num = len(imgs_cam)

        if identified_disease_num == 0:
            print('%s is normal'%img_path)
            continue
        else:
            imgs = [[org_img, imgs_cam[i], imgs_gradcam[i], imgs_gradcampp[i]]for i in range(identified_disease_num)]
        
        plt.figure(figsize=(8, 4))
        plt.clf()
        
        i = 1
        tool_name = ['org', 'cam', 'gradcam', 'gradcampp']
        for img_row, idx in zip(imgs, idx1):
            for img, t_name in zip(img_row, tool_name):
                plt.subplot(identified_disease_num, 4, i)
                plt.title(t_name)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                i += 1

        disease = ''
        for i in idx1:
            disease += str(i.cpu().int().numpy())

        plt.show()
        plt.savefig('./example_results/%s_%s.png'%(img_name, disease))
    
if __name__ == '__main__':

    # main('gradcam')
    # main('gradcampp')
    # main('cam')
    main()


def display(args, display_method):

    arch = Model(8, 'densenet').to(args.device)
    arch.load_state_dict(torch.load('./save/best_48_large.pkl'))
    arch.eval()

    if display_method == 'gradcam' or 'guided-gradcam':
        tool = GradCAM(arch)
    elif display_method == 'gradcampp':
        tool = GradCAMpp(arch)
    elif display_method == 'cam':
        tool = CAM(arch)

    transforms, display_transforms = get_transforms()

    img_path = '/mnt/data1/MedicalDataset/Kaggle/valid/9639_left.jpeg'
    img = Image.open(img_path).convert('RGB')

    im = transforms(img).to(args.device)
    img = display_transforms(img).to(args.device)

    im = im.unsqueeze(0)
    normal, mask, _ = tool(im)

    if display_method == 'guided-gradcam':
        GBP = GuidedBackprop(arch)
        guided_mask = GBP.generate_gradients(im, 0, args.device)
        mask1 = mask[0].squeeze().unsqueeze(0)

    save_gradient_images(mask1, 'Guided_GradCAM')

    if not normal:
        heatmaps, results = visualize_cam(mask, img)

        for i, result in enumerate(results):

            img = T.ToPILImage()(result)
            img.save('./cam_imgs/%s_%d_2.png'%(display_method, i))

    else:
        print('Normal')