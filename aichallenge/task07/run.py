import os
import sys
import random
import datetime
import argparse
import xml.etree.ElementTree as ET 

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.baseline import get_model
from models.tools.split_dataset import train_valid_split, xml_parser, xml_writer, get_test_list, indent
from models.tools.custom_dataset import CustomDataset
from models.tools.augmentation import Compose, ToTensor, Resize, Normalize
from models.tools.utils import AverageMeter
from models.tools.evaluation import evaluation_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--num_classes', type=int, default=29)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--print_result', type=int, default=10)
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()

device = 'cuda:0' if args.cuda else 'cpu'

class_nums = {
    1: 'bus', 2: 'car', 3:'carrier', 4: 'cat',  5: 'dog', 
    6: 'motorcycle', 7: 'movable_signage', 8: 'person', 9: 'scooter', 10: 'stroller', 
    11: 'truck', 12: 'wheelchair', 13: 'barricade', 14: 'bench', 15: 'chair',
    16: 'fire_hydrant', 17: 'kiosk', 18: 'parking_meter', 19: 'pole', 20: 'potted_plant', 
    21: 'power_controller', 22: 'stop', 23: 'table', 24: 'traffic_light_controller', 
    25: 'traffic_sign', 26: 'tree_trunk', 27: 'bollard', 28: 'bicycle'
}
class_names = {v: k for k, v in class_nums.items()}

if not os.path.exists('./input/train_labels.xml'):
    train_list, valid_list = train_valid_split(test_size=0.5, shuffle=True)
    train_objects, _ = xml_parser(train_list, valid_list)
    xml_writer(train_objects, 'train')
    # xml_writer(valid_objects, 'valid')
    print('Generated labels.xml...')

train_augmentations = Compose([
    Resize((500,285)), # (400, 225), (500, 285)
    Normalize(),
    ToTensor(),
])
test_augmentations = Compose([
    Normalize(),
    ToTensor(),
])

if args.mode != 'test':
    trainset = CustomDataset(class_names=class_names, augmentation=train_augmentations, phase='train')
    # validset = CustomDataset(class_names=class_names, augmentation=train_augmentations, phase='valid')
    # print(f'Size: trainset: {len(trainset):,}, validset: {len(validset):,}, testset: {len(testset):,}')
    # print(f'Cleaned dataset size: {len(trainset)+len(validset):,}')

testset = CustomDataset(class_names=class_names, augmentation=test_augmentations, phase='test')

def collate_fn(batch):
    return tuple(zip(*batch))


if args.mode != 'test':
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # valid_loader = DataLoader(validset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def train_fn(model, data_loader, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    start_time = datetime.datetime.now()
    num_images = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        num_images += len(images)
        images = torch.stack(images)
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        losses.update(loss.item(), images.size(0))
        
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_result == 0:
            print(f'Epoch {epoch+1}[{len(data_loader.dataset):,}({num_images/len(data_loader.dataset):.2f})] '
                  f'- Elapsed time: {datetime.datetime.now() - start_time}\n'
                  f' - loss: classifier={loss_dict["loss_classifier"]:.6f}, box_reg={loss_dict["loss_box_reg"]:.6f}, '
                  f'objectness={loss_dict["loss_objectness"]:.6f}, rpn_box_reg={loss_dict["loss_rpn_box_reg"]:.6f}')
    
        
def valid_fn(model, data_loader, device, class_names, valid=True):
    model.eval()
    xml_root = ET.Element('predictions')
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(data_loader)):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for j, output in enumerate(outputs):
                image_name = data_loader.dataset.images[i*batch_size+j]
                xml_image = ET.SubElement(xml_root, 'image', {'name': image_name})

                boxes = output['boxes'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    attribs = {
                        'class_name': class_names[label],
                        'score': str(float(score)), 
                        'x1': str(int(box[0])), 
                        'y1': str(int(box[1])), 
                        'x2': str(int(box[2])), 
                        'y2': str(int(box[3]))
                    }
                    ET.SubElement(xml_image, 'predict', attribs)
                
    indent(xml_root)
    tree = ET.ElementTree(xml_root)
    if valid:
        tree.write('./output/validation.xml')
    else:
        tree.write('./output/prediction.xml')
    print('Save predicted labels.xml...\n')


def get_valid(name):
    path_gt = os.path.join('./input/valid_labels.xml')
    path_dr = os.path.join('./output/validation.xml')
    result = evaluation_metrics(path_gt, path_dr)
    print('validation : ', result)


def fixed_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


fixed_seed(42)
if args.mode == 'train':
    model = get_model(args.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        train_fn(model, train_loader, optimizer, device, epoch)
        torch.save(model.state_dict(), f'./weight/test-last{epoch+1}.pth')
else:
    print('Loading model...')
    model = get_model(args.num_classes).to(device)
    model.load_state_dict(torch.load('./weight/test-last1.pth'))
    model.eval()
    print('Loaded model...')
    # pass

# valid_fn(model, valid_loader, device, class_nums, valid=True)
# get_valid('validation')
valid_fn(model, test_loader, device, class_nums, valid=False)
# get_valid('prediction')
