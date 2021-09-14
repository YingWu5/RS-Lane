import os
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2
import ujson as json
# from .video import video2frame
import random
import PIL


VGG_MEAN = [103.939, 116.779, 123.68]


class TuSimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_dir, phase, size=(512,288), transform=None):
        """
        Args:
            dataset_dir: The directory path of the dataset
            phase: 'train', 'val', or 'test'
        """
        self.dataset_dir = dataset_dir
        self.phase = phase
        self.size = size
        self.transform = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.6,contrast=0.3,saturation=0.3,hue=0.2),
        ]),p=0.5)
        self.albumentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomSizedCrop(min_max_height=(180,288),height=288,width=512,w2h_ratio=1.7,p=0.7),
        ])
        assert os.path.exists(dataset_dir), 'Directory {} does not exist!'.format(dataset_dir)

        if phase == 'train' or phase == 'val':
            label_files = list()
            if phase == 'train':
                label_files.append(os.path.join(dataset_dir, 'label_data_0313.json'))
                label_files.append(os.path.join(dataset_dir, 'label_data_0601.json'))
            elif phase == 'val':
                label_files.append(os.path.join(dataset_dir, 'label_data_0531.json'))

            self.image_list = []
            self.lanes_list = []
            for file in label_files:
                try:
                    for line in open(file).readlines():
                        info_dict = json.loads(line)
                        self.image_list.append(info_dict['raw_file'])

                        h_samples = info_dict['h_samples']
                        lanes = info_dict['lanes']

                        xy_list = list()
                        for lane in lanes:
                            y = np.array([h_samples]).T
                            x = np.array([lane]).T
                            xy = np.hstack((x, y))

                            index = np.where(xy[:, 0] > 2)
                            xy_list.append(xy[index])
                        self.lanes_list.append(xy_list)
                except BaseException:
                    raise Exception(f'Fail to load {file}.')

        elif phase == 'test':
            task_file = os.path.join(dataset_dir, 'test_tasks_0627.json')
            try:
                self.image_list = [json.loads(line)['raw_file'] for line in open(task_file).readlines()]
            except BaseException:
                raise Exception(f'Fail to load {task_file}.')

        else:
            raise Exception(f"Phase '{self.phase}' cannot be recognize!")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        if self.phase == 'train' or self.phase == 'val':

            '''OpenCV'''
            img_path = os.path.join(self.dataset_dir, self.image_list[idx])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w, c = image.shape
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            
            bin_seg_label = np.zeros((720, 1280), dtype=np.uint8)
            inst_seg_label = np.zeros((720, 1280), dtype=np.uint8)

            lanes = self.lanes_list[idx]
            for idx, lane in enumerate(lanes):
                cv2.polylines(bin_seg_label, [lane], False, 1, 5)
                cv2.polylines(inst_seg_label, [lane], False, idx+1, 15)  

            bin_seg_label = cv2.resize(bin_seg_label, self.size, interpolation=cv2.INTER_NEAREST)
            inst_seg_label = cv2.resize(inst_seg_label, self.size, interpolation=cv2.INTER_NEAREST)

            masks = [bin_seg_label,inst_seg_label]
            transformed = self.albumentations(image=image, masks=masks)
            image = transformed['image']
            bin_seg_label,inst_seg_label = transformed['masks']

            bin_seg_label = torch.from_numpy(bin_seg_label).long()
            inst_seg_label = torch.from_numpy(inst_seg_label).long()

            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()/255

            image = self.transform(image).float()

            sample = {'input_tensor': image, 'binary_tensor': bin_seg_label, 'instance_tensor': inst_seg_label,
                      'raw_file':self.image_list[idx]}


            return sample

        elif self.phase == 'test':
            '''OpenCV'''
            img_path = os.path.join(self.dataset_dir, self.image_list[idx])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w, c = image.shape
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255

            clip, seq, frame = self.image_list[idx].split('/')[-3:]
            path = '/'.join([clip, seq, frame])
            orignal_size=(h,w)

            sample = {'input_tensor': image, 'raw_file':self.image_list[idx], 'path':path,
                'orignal_size':orignal_size}

            return sample
        
        elif self.phase == 'video':
            img_path = self.image_list[idx]
            ori_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w, c = ori_image.shape
            image = cv2.resize(ori_image, self.size, interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255

            video, frame = self.image_list[idx].split('/')[-2:]
            frame=os.path.splitext(frame)[0]
            video=video.split('_frames')[0]
            path = '/'.join([video, frame])
            orignal_size=(h,w)

            sample = {'input_tensor': image, 'raw_file':self.image_list[idx], 'path':path,
                'orignal_size':orignal_size}

            return sample

        else:
            raise Exception(f"Phase '{self.phase}' cannot be recognize!")


class CULane(Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()


    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))

        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def createIndex_test(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))

        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, line[1:]))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx],cv2.IMREAD_COLOR)
        img = cv2.resize(img, (800,288), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()/255
        if self.image_set != 'test':
            inst_seg_label = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            inst_seg_label = cv2.resize(inst_seg_label, (800,288), interpolation=cv2.INTER_LINEAR)
            _, bin_seg_label = thresh1 = cv2.threshold(inst_seg_label,0,1,cv2.THRESH_BINARY)
            bin_seg_label = torch.from_numpy(bin_seg_label).long()
            inst_seg_label = torch.from_numpy(inst_seg_label).long()
            sample = {'input_tensor': img,
                  'instance_tensor': inst_seg_label,
                  'binary_tensor': bin_seg_label,
                  'raw_file': self.img_list[idx]}
        else:
            sample = {'input_tensor': img,
                  'raw_file': self.img_list[idx]}

        
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    
    train_set = CULane(path = 'CULane', image_set='train')

    for idx, item in enumerate(train_set):

        input_tensor = item['input_tensor']
        bin_seg_label = item['binary_tensor']
        inst_seg_label = item['instance_tensor']
        raw_file = item['raw_file'] 

        input = ((input_tensor*255).numpy().transpose(1, 2, 0) ).astype(np.uint8)
        bin_seg_label = (bin_seg_label * 255).numpy().astype(np.uint8)
        inst_seg_label = (inst_seg_label * 40).numpy().astype(np.uint8)

        cv2.imshow('input', input)
        cv2.imshow('bin_seg_label', bin_seg_label)
        cv2.imshow('inst_seg_label', inst_seg_label)

        print(raw_file)
       
        cv2.waitKey(0)
        cv2.destroyAllWindows()

       

