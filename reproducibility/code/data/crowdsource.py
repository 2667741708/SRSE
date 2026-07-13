import json
import numpy as np
import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset
import wandb

def get_dataset(args, transform_train, transform_test):
    
    if args.slice == 1:
        train_split, test_split = ['fold1', 'fold2', 'fold4', 'fold5'], ['fold3']
    elif args.slice == 2:
        train_split, test_split = ['fold1','fold2','fold5'],['fold4']
    elif args.slice == 3:
        train_split, test_split = ['fold1','fold2','fold3'],['fold5']
    
    
    trainset = Crowdsource(args, splits=train_split, transform=transform_train)
    trainset.partial_noise() 
    testset = Crowdsource(args, splits=test_split, transform=transform_test)
    return trainset, testset

class Crowdsource(Dataset):

    def __init__(self, args, splits=['fold1'], transform=None):
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.args = args
        self.num_classes = self.args.num_classes
        self.noisy_labels = []
        self.train = len(splits) != 1

        
        annotation_file = os.path.join(self.root, 'annotations.json')
        if not os.path.exists(annotation_file):
            
            
            potential_path = os.path.join(self.root, args.dataset, 'annotations.json')
            if os.path.exists(potential_path):
                self.root = os.path.join(self.root, args.dataset)
                annotation_file = potential_path
        
        print(f"Loading annotations from: {annotation_file}")
        with open(annotation_file, 'r') as outfile:
            annotation_jsons = json.load(outfile)
            
        
        if isinstance(annotation_jsons, list):
            raw_entries = annotation_jsons[0]["annotations"]
        else:
            raw_entries = annotation_jsons["annotations"]

        
        img_names_set = set()
        unique_labels_set = set()
        
        for entry in raw_entries:
            if entry["class_label"] is not None:
                img_names_set.add(entry["image_path"])
                unique_labels_set.add(entry["class_label"])
                
        img_names = sorted(list(img_names_set))
        
        
        try:
            unique_labels = sorted(list(unique_labels_set), key=lambda x: int(x))
        except:
            unique_labels = sorted(list(unique_labels_set))
            
        # label_to_idx: {'head_injury': 0, 'not_injured': 1, ...}
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        
        if len(unique_labels) != self.num_classes:
            print(f"Warning: args.num_classes={self.num_classes} but found {len(unique_labels)} unique labels in JSON.")
            print(f"Please update your --num_classes_map in the training script to {len(unique_labels)}!")

        
        map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
        
        
        self.folds = {}
        for name in img_names:
            
            fold = name.split('/')[1] if '/' in name else name.split('\\')[1]
            if fold in self.folds:
                self.folds[fold].append(map_names[name])
            else:
                self.folds[fold] = [map_names[name]]
        
        
        
        _data = np.zeros((len(img_names), self.num_classes))

        for entry in raw_entries:
            if entry["class_label"] is not None:
                img_idx = map_names[entry["image_path"]]
                label_str = entry["class_label"]
                
                
                if label_str in self.label_to_idx:
                    label_idx = self.label_to_idx[label_str]
                    if label_idx < self.num_classes:
                        _data[img_idx, label_idx] += 1
                    else:
                        pass 

        
        data = np.zeros((len(img_names), self.num_classes))
        rng = np.random.default_rng(self.args.seed_dataset)
        
        for i in range(data.shape[0]):
            total_votes = _data[i].sum()
            if total_votes > 0:
                probs = _data[i] / total_votes
                
                annots = rng.choice(self.num_classes, p=probs, size=args.lpi)
                for a in annots:
                    data[i, a] += 1
            else:
                
                pass
                
        partialY = np.zeros((len(img_names), self.num_classes))
        partialY[data.nonzero()] = 1
        
        weights = data / (data.sum(1, keepdims=True) + 1e-8)
        clean_labels = np.argmax(_data, axis=1)
        
        
        req_ids = []
        for split in splits:
            if split in self.folds:
                req_ids.extend(self.folds[split])
            else:
                print(f"Warning: Split {split} not found (Available: {list(self.folds.keys())})")

        self.soft_labels = partialY[req_ids]
        self.clean_labels = clean_labels[req_ids]
        self.weights = weights[req_ids]
        self.targets = np.copy(self.clean_labels)
        
        
        self.data = []
        for i in req_ids:
            relative_path = img_names[i]
            full_path = os.path.join(self.root, relative_path)
            
            
            if not os.path.exists(full_path):
                
                parent_root = os.path.dirname(self.root.rstrip(os.sep))
                alt_path = os.path.join(parent_root, relative_path)
                if os.path.exists(alt_path):
                    full_path = alt_path
            
            self.data.append(full_path)

        
        majority_label = np.argmax(self.weights, axis=1)
        clean_majority = sum(majority_label == self.clean_labels)
        
    def partial_noise(self):
        self.targets = np.zeros((len(self.clean_labels),)) - 1

    def __getitem__(self, index):
        img_path, labels = self.data[index], self.targets[index]
        
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        if self.train:
            return img, labels, index
        else:
            return img, labels
        
    def __len__(self):
        return len(self.targets)    
