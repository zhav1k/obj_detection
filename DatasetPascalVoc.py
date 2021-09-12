import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class PascalVocDatasetClass(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))
        '''
        coco_annotation has the following structure:
        [{'id': 1, 'image_id': 1, 'category_id': 15, 'bbox': [269, 90, 150, 284], 'area': 42600, 'segmentation': [], 'iscrowd': 0}]
        '''
        num_objs = len(coco_annotation)
        boxes = []
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for i in range(num_objs):
            labels.append(coco_annotation[i]['category_id'])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = []
        for i in range(num_objs):
            iscrowds.append(coco_annotation[i]['iscrowd'])
        iscrowds = torch.tensor(iscrowds, dtype=torch.int64)

        # Annotation is in dictionary format
        new_annotation = {}
        new_annotation["boxes"] = boxes
        new_annotation["labels"] = labels
        new_annotation["image_id"] = img_id
        new_annotation["area"] = areas
        new_annotation["iscrowd"] = iscrowds

        if self.transforms is not None:
            img = self.transforms(img)

        return img, new_annotation

    def __len__(self):
        return len(self.ids)