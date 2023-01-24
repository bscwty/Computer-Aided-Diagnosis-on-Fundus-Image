import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import copy
import numpy as np
from PIL import Image

from seg.models import LadderNet
from seg.config import parse_args
from seg.lib.common import setpu_seed
from seg.lib.dataset import TestDataset
from seg.lib.extract_patches import get_one_img_testset, recompone_overlap


setpu_seed(2021)

class Test():
    def __init__(self, args, img, device):
        self.args = args
        # save path
        self.device = device

        self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_one_img_testset(
            img,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )

        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)

        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        result = self.pred_imgs[0]

        result[result >= 0.5] = 1
        result[result <  0.5] = 0  

        result = result.squeeze() * 255

        return result
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        return 

class Seg:

    def __init__(self, vess_seg_paht, disc_seg_path) -> None:

        self.args = parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        self.vess_seg = vess_seg_paht
        self.disc_seg = disc_seg_path

    def init_net(self):

        cudnn.benchmark = True

        self.vess_seg_net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(self.device)
        self.disc_seg_net = copy.deepcopy(self.vess_seg_net)

        '''
        TODO
        '''

        # self.vess_seg_net.load_state_dict(torch.load(self.vess_seg, map_location='cpu')['net'])
        # self.disc_seg_net.load_state_dict(torch.load(self.disc_seg, map_location='cpu')['net'])

        self.vess_seg_net.load_state_dict(torch.load(self.vess_seg)['net'])
        self.disc_seg_net.load_state_dict(torch.load(self.disc_seg)['net'])

        self.vess_seg_net = self.vess_seg_net.to(self.device)
        self.disc_seg_net = self.disc_seg_net.to(self.device)

        return

    def get_result(self, img: Image.Image) -> Image.Image:

        eval = Test(self.args, img, self.device)

        vess_result = eval.inference(self.vess_seg_net)[np.newaxis,:,:]
        disc_result = eval.inference(self.disc_seg_net)[np.newaxis,:,:]

        result = np.vstack([disc_result, vess_result, np.zeros_like(vess_result, dtype=np.int8)])
        result = result.transpose(1, 2, 0).astype(np.uint8)
        result = Image.fromarray(result)

        return result

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.get_result(img)


if __name__ == '__main__':

    import time

    img = './A.jpg'
    img = Image.open(img).convert('RGB')

    seg = Seg()

    start = time.time()
    result = seg.get_result(img)
    end = time.time()

    print(end-start)