
import random
from typing import List, Dict
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size

    def resize(self, image, target):
        w, h = image.size
        image = self._letterbox_image(image, self.size)
        scale = min(self.size[0]/h, self.size[1]/w)

        target['boxes'][:,:4] *= scale

        new_w = scale * w
        new_h = scale * h
        inp_dim = self.size

        del_w = (self.size[1] - new_w)/2
        del_h = (self.size[0] - new_h)/2
        
        
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
        target['boxes'][:,:4] += add_matrix
        # print('Converted', target['boxes'])

        # scale = max(self.size[0]/h, self.size[1]/w)
        # target['boxes'][:,:4] -= add_matrix
        # target['boxes'][:,:4] /= scale
        

        # print('Decoding', target['boxes'][:,:4])

        # im = np.array(image, dtype=np.uint8)
        # plt.imshow(image)

        image = image.astype(np.uint8)
        
        return image, target

    def _letterbox_image(self, image, size):
        img_w, img_h = image.size
        w, h = size
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))

        resized_image = image.resize((new_w, new_h), Image.ANTIALIAS)
        # print('Resized image', resized_image)
        #create a black canvas    
        canvas = np.full((w, h, 3), 128)
        # print('Canvas size', canvas.shape)
    
        #paste the image on the canvas
        fh = (h-new_h)//2
        fw = (w-new_w)//2
        canvas[fh:fh+new_h, fw:fw+new_w,  :] = resized_image
        # print('Canvas size', canvas.shape)
        
        return canvas


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target




# class Resize:
#     """
#     레터박스 추가
#     """
#     def __init__(self, size: Tuple[int, int]):
#         self.size = size

#     def __call__(self, image: PIL.Image, target: Dict[str, torch.Tensor]):
#         w, h = image.size
#         image = self._letterbox_image(image, self.size)
#         scale = min(self.size[0]/h, self.size[1]/w)
#         target['boxes'][:,:4] *= scale

#         new_w = scale * w
#         new_h = scale * h

#         del_w = (self.size[1] - new_w)/2
#         del_h = (self.size[0] - new_h)/2
        
#         add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
#         target['boxes'][:,:4] += add_matrix

#         image = image.astype(np.uint8)

#         return image, target

#     def _letterbox_image(self, image: PIL.Image, size: Tuple[int, int]) -> np.ndarray:
#         img_w, img_h = image.size
#         w, h = size
#         new_w = int(img_w * min(w/img_w, h/img_h))
#         new_h = int(img_h * min(w/img_w, h/img_h))

#         resized_image = image.resize((new_w, new_h), Image.ANTIALIAS)
#         canvas = np.full((w, h, 3), 128)

#         fh = (h-new_h)//2
#         fw = (w-new_w)//2
#         canvas[fh:fh+new_h, fw:fw+new_w,  :] = resized_image
        
#         return canvas

