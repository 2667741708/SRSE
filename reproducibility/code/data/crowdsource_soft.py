import json
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import wandb

def get_crowdsourced_dataset(args, transform_train, transform_test):
    if args.slice == 1: train_split, test_split = ['fold1','fold4','fold5'],['fold3']
    elif args.slice == 2: train_split, test_split = ['fold1','fold2','fold5'],['fold4']
    else: train_split, test_split = ['fold1','fold2','fold3'],['fold5']
    trainset = CrowdsourcedDataset(args, splits=train_split, transform=transform_train, train=True)
    testset = CrowdsourcedDataset(args, splits=test_split, transform=transform_test, train=False)
    return trainset, testset

class CrowdsourcedDataset(Dataset):
    def __init__(self, args, splits, transform=None, train=True):
        # self.root = os.path.join(args.train_root, args.dataset)
        # 处理 Treeversity 数据集的特殊文件夹名称
        dataset_folder_name = args.dataset
        if dataset_folder_name == 'Treeversity':
            dataset_folder_name = 'Treeversity#6'
            
        self.root = os.path.join(args.train_root, dataset_folder_name)        
        self.transform = transform
        self.args = args
        self.train = train
        self.num_classes = args.num_classes
        
        annotation_file = os.path.join(self.root, 'annotations.json')
        with open(annotation_file, 'r') as f: annotations_json = json.load(f)

        all_img_paths_raw = sorted(list(set(entry["image_path"] for entry in annotations_json[0]["annotations"] if entry["class_label"] is not None)))
        
        # # [关键步骤] 清理路径，去掉数据集名称前缀，使其成为相对于 self.root 的路径
        # dataset_folder_name = os.path.basename(self.root) # e.g., "Plankton"
        # all_img_paths = []
        # for path in all_img_paths_raw:
        #     if path.startswith(dataset_folder_name + '/'):
        #         all_img_paths.append(path[len(dataset_folder_name)+1:])
        #     else:
        #         all_img_paths.append(path)
        # [关键步骤] 清理路径，去掉数据集名称前缀，使其成为相对于 self.root 的路径
        dataset_folder_name_with_hash = os.path.basename(self.root) # e.g., "Treeversity#6"
        dataset_folder_name_without_hash = dataset_folder_name_with_hash.split('#')[0] # e.g., "Treeversity"

        all_img_paths = []
        for path in all_img_paths_raw:
            # 优先尝试匹配带 # 的名称, e.g., "Treeversity#6/"
            if path.startswith(dataset_folder_name_with_hash + '/'):
                all_img_paths.append(path[len(dataset_folder_name_with_hash)+1:])
            # 如果不匹配，再尝试匹配不带 # 的名称, e.g., "Treeversity/"
            elif path.startswith(dataset_folder_name_without_hash + '/'):
                all_img_paths.append(path[len(dataset_folder_name_without_hash)+1:])
            # 如果都不匹配，则假定路径已经是相对路径
            else:
                all_img_paths.append(path)        
        all_labels = sorted(list(set(entry["class_label"] for entry in annotations_json[0]["annotations"] if entry["class_label"] is not None)))
        
        # path_to_idx 使用清理后的路径作为键
        path_to_idx = {path: i for i, path in enumerate(all_img_paths)}; label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        if self.num_classes != len(all_labels):
            print(f"Warning: --num_classes is {self.num_classes}, but found {len(all_labels)} labels. Using {len(all_labels)}.")
            self.num_classes = len(all_labels); args.num_classes = len(all_labels)

        vote_matrix = np.zeros((len(all_img_paths), self.num_classes))
        
        # [第一次修复] 填充 vote_matrix 时，也要使用清理后的路径来查找索引
        for entry in annotations_json[0]["annotations"]:
            if entry["class_label"] is not None:
                original_path = entry["image_path"]
                cleaned_path = original_path
                if original_path.startswith(dataset_folder_name + '/'):
                    cleaned_path = original_path[len(dataset_folder_name)+1:]
                
                # 确保 cleaned_path 存在于字典中
                if cleaned_path in path_to_idx:
                    vote_matrix[path_to_idx[cleaned_path], label_to_idx[entry["class_label"]]] += 1

        if self.train and hasattr(args, 'lpi') and args.lpi > 0:
            simulated_votes = np.zeros_like(vote_matrix)
            rng = np.random.default_rng(self.args.seed_dataset)
            for i in range(len(all_img_paths)):
                if vote_matrix[i].sum() > 0:
                    # probs = vote_matrix[i] / vote_matrix[i].sum()
                    #more harder to learn
                    raw_probs = vote_matrix[i] / vote_matrix[i].sum()

                    # 使用一个温度参数 "T" 来平滑概率分布
                    # T > 1 会使分布更平坦, T < 1 会使分布更尖锐
                    temperature = 2.0 
                    smoothed_probs = np.power(raw_probs, 1.0 / temperature)
                    probs = smoothed_probs / smoothed_probs.sum()

                    annots = rng.choice(self.num_classes, p=probs, size=args.lpi)                    
                    annots = rng.choice(self.num_classes, p=probs, size=args.lpi)
                    for a in annots: simulated_votes[i, a] += 1
            final_votes = simulated_votes
        else: final_votes = vote_matrix

        # self.soft_labels = (final_votes > 0).astype(float); self.clean_labels = np.argmax(vote_matrix, axis=1)
        self.soft_labels = (final_votes > 0).astype(np.float32); self.clean_labels = np.argmax(vote_matrix, axis=1)

        fold_indices = {}
        for path, idx in path_to_idx.items():
            # [第二次修复] 因为 'path' 是清理后的路径 (如 'fold1/...'), 所以 fold 在索引 0
            fold = path.split('/')[0]
            if fold not in fold_indices: fold_indices[fold] = []
            fold_indices[fold].append(idx)
            
        req_ids = sorted(list(set(sum([fold_indices.get(s, []) for s in splits], []))))

        self.data_paths = [all_img_paths[i] for i in req_ids]
        self.soft_labels = self.soft_labels[req_ids]; self.clean_labels = self.clean_labels[req_ids]
        
        if self.train: self.targets = np.full(len(self.clean_labels), -1, dtype=np.int64)
        else: self.targets = self.clean_labels
        
        if self.train and len(self.data_paths) > 0:
            self._log_stats()
        elif self.train:
             print("Warning: Loaded 0 training samples after applying splits. Check your data and fold names.")


    def _log_stats(self):
        # 增加一个检查，防止 ZeroDivisionError
        if len(self.clean_labels) == 0:
            print("Error: No labels found in the training set after splitting.")
            return

        avg_candidates = self.soft_labels.sum(1).mean()
        clean_in_partial = sum(self.soft_labels[np.arange(len(self.clean_labels)), self.clean_labels] == 1)
        print(f"Loaded {len(self.data_paths)} training samples for {self.args.dataset}.")
        print(f"  -> Average candidate num: {avg_candidates:.2f}")
        print(f"  -> True label is in candidate set for {clean_in_partial}/{len(self.clean_labels)} samples ({clean_in_partial/len(self.clean_labels):.2%}).")

    def __len__(self): return len(self.data_paths)

    def __getitem__(self, index):
        # 使用健壮的路径拼接方式
        img_path = os.path.join(self.root, self.data_paths[index])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None: img = self.transform(img)
            
        if self.train: return img, -1, index
        else: return img, self.targets[index]