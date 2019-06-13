import torch
from torch.utils import data
import os
from PIL import Image
from transform import resize, random_flip, random_crop, center_crop

class VocDataset(data.Dataset):
    def __init__(self, root , train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images and annotations.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []
        from glob import glob

        annotation_files=glob(os.path.join(root,'*.xml'))

        import parse_voc_annotation
        for annotation_file in annotation_files:
            # print(annotation_file)
            try:
                annotation_dict=parse_voc_annotation.parse_voc_to_dict(annotation_file)
                filename,classes_text, box=parse_voc_annotation.voc_format_to_object_detect_format(annotation_dict)
                if len(box)>0:
                    self.boxes.append(torch.Tensor(box))
                    # because the annotations are not precious, so i use 0 to override
                    self.labels.append(torch.LongTensor([0]*len(classes_text)))
                    self.fnames.append(filename)
                else:
                    print("error annotatiom found",filename)
            except:
                # print("error encounter")
                i=0
        self.num_samples = len(self.fnames)
        print("Dataset init completely,total dataset num is ",self.num_samples)
    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def __len__(self):
        return self.num_samples





if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = VocDataset(root=r'H:\drone_image_and_annotation_mixed\test',
                         train=True, transform=transform,
                         input_size=1024)

    print((dataset[153][1][0]))

