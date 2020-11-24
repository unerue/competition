import os
import sys
from glob import glob
import xml.etree.ElementTree as ET
from typing import List, Union
import numpy as np
from PIL import Image
from tqdm import tqdm


def train_valid_split(
    path: str = './data/train/', test_size: float = 0.1, shuffle: bool = True) -> List:
    """
    베이스라인 코드가 정말 별로라 새로 데이터 분할 후 .xml 파일 생성
    """
    all_files = list(glob(path + '*jpg')) + list(glob(path + '*png'))
    print(f'Total dataset: {len(all_files)}...')

    split_index = int(len(all_files)*(1-test_size))
    if shuffle:
        shuffled_index = np.random.permutation(len(all_files))
    else:
        shuffled_index = np.arange(len(all_files))
        
    train_index = shuffled_index[:split_index]
    valid_index = shuffled_index[split_index:]
    all_files = np.asarray(all_files)
    
    return list(all_files[train_index]), list(all_files[valid_index])
   

def xml_parser(
    train_list: List[str], valid_list: List[str]) -> ET.XML:
    """ 
    대회 서버 내 데이터셋 레이블 파일 문제로 인해 예외처리해야 함
    """
    label_file_name = 'train.xml'
    root = ET.parse(f'./data/train/{label_file_name}').getroot()
    print('Parsed all.xml...')

    train_list: List[str] = [f.rsplit('/',1)[1] for f in train_list]
    valid_list: List[str] = [f.rsplit('/',1)[1] for f in valid_list]
    train_objects: List = []
    valid_objects: List = []
    error_train_files: List[str] = []
    error_valid_files: List[str] = []
    for child in tqdm(root):
        if child.tag not in ['version', 'meta']:
            file_name: str = child.get('name')
            for attrib in child:
                try:
                    attrib.attrib['xtl']
                    attrib.attrib['xbr']
                    attrib.attrib['ytl']
                    attrib.attrib['ybr']
                    Image.open('./data/train/' + file_name)
                    Image.open('./data/train/' + file_name)
                except (KeyError, SyntaxError, IOError, EOFError, ValueError, FileNotFoundError) as e:
                    print(e, file_name)
                    if file_name in train_list:
                        error_train_files.append(file_name)
                        train_list.remove(file_name)
                    elif file_name in valid_list:
                        error_valid_files.append(file_name)
                        valid_list.remove(file_name)

            if file_name in train_list and file_name not in error_train_files:
                train_objects.append(child)
            elif file_name in valid_list and file_name not in error_valid_files:
                valid_objects.append(child)
    
    return train_objects, valid_objects


def indent(node: ET.XML, level: int = 0):
    i = '\n' + level * ' ' * 4
    if len(node):
        if not node.text or not node.text.strip():
            node.text = i + ' ' * 4
        if not node.tail or not node.tail.strip():
            node.tail = i
        for node in node:
            indent(node, level + 1)
        if not node.tail or not node.tail.strip():
            node.tail = i
    else:
        if level and (not node.tail or not node.tail.strip()):
            node.tail = i


def xml_writer(objects: List, file_name: str):
    if len(objects) == 0:
        return 

    root = ET.Element('annotations')
    for obj in objects:
        image = ET.SubElement(root, 'image', obj.attrib)
        for bbox in obj:
            ET.SubElement(image, 'box', bbox.attrib)

    indent(root)
    tree = ET.ElementTree(root)
    tree.write(f'./data/{file_name}_labels.xml')


def get_test_list(path: str = './data/test/') -> List[str]:
    test_files: List[str] = sorted(list(glob(path + '*jpg')) + list(glob(path + '*png')))
    print(f'Total testset: {len(test_files)}...')
    return test_files


if __name__ == '__main__' :
    train_list, valid_list = train_valid_split(test_size=0.2, shuffle=True)
    train_objects, valid_objects = xml_parser(train_list, valid_list)
    xml_writer(train_objects, 'train')
    xml_writer(valid_objects, 'valid')


