# import json
# import numpy as np
# import tqdm
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# import wandb

# def get_dataset(args, transform_train, transform_test):
#     if args.slice == 1:
#         # train_split, test_split = ['fold1','fold4','fold5'],['fold3']
#         train_split, test_split = ['fold1', 'fold2', 'fold4', 'fold5'], ['fold3']
#     elif args.slice == 2:
#         train_split, test_split = ['fold1','fold2','fold5'],['fold4']
#     elif args.slice == 3:
#         train_split, test_split = ['fold1','fold2','fold3'],['fold5']
#     trainset = Crowdsource(args, splits=train_split, transform=transform_train)
#     trainset.partial_noise()
#     testset = Crowdsource(args, splits=test_split, transform=transform_test)
#     return trainset, testset

# class Crowdsource(Dataset):

#     def __init__(self, args, splits=['fold1'], transform=None):
#         self.root = os.path.expanduser(args.train_root)
#         # self.root = './Treeversity#6'
#         self.transform = transform
#         self.args = args
#         self.num_classes = self.args.num_classes
#         self.noisy_labels = []
#         self.train = len(splits)!=1

#         # annotation_file = self.root + '/annotations.json'
#         annotation_file = os.path.join(self.root, 'annotations.json')
#         with open(annotation_file, 'r') as outfile:
#             annotation_jsons = json.load(outfile)
            
#         img_names = []
#         labels = []
#         for entry in annotation_jsons[0]["annotations"]:
#         # add only valid annotations to table
#             if entry["class_label"] is not None:
#                 # if entry["image_path"] not in img_names:
#                 img_names.append(entry["image_path"])
#                 # if entry["class_label"] not in img_names:
#                 labels.append(entry["class_label"])
                
#         img_names = list(np.unique(np.array(img_names)))
#         labels = list(np.unique(np.array(labels)))

#         # fast access maps
#         map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
#         map_labels = dict(zip(labels, list(np.arange(0, len(labels)))))
        
#         self.folds = {}
        
#         for name in img_names:
#             fold = name.split('/')[1]
#             if fold in self.folds:
#                 self.folds[fold].append(map_names[name])
#             else:
#                 self.folds[fold] = [map_names[name]]
        
#         _data = np.zeros((len(img_names), len(labels)))

#         for entry in annotation_jsons[0]["annotations"]:
#             # add only valid annotations to table
#             if entry["class_label"] is not None:
#                 _data[map_names[entry["image_path"]], map_labels[entry["class_label"]]] += 1

#         data = np.zeros((len(img_names), len(labels)))
#         rng = np.random.default_rng(self.args.seed_dataset)
#         for i in range(data.shape[0]):
#             annots = rng.choice(args.num_classes,p=_data[i]/_data[i].sum(),size=args.lpi)
#             for a in annots:
#                 data[i,a] += 1
#             assert data[i].sum() == args.lpi
                
                
#         partialY = np.zeros((len(img_names), len(labels)))
#         partialY[data.nonzero()] = 1
        
#         weights = data/data.sum(1,keepdims=True)
#         clean_labels = np.argmax(_data,axis=1)
        
#         req_ids = []
#         for split in splits:
#             req_ids.extend(self.folds[split])
        

#         self.soft_labels = partialY[req_ids]
#         self.clean_labels = clean_labels[req_ids]
#         # self.data = [img_names[i] for i in req_ids]
#         self.data = []
#         for i in req_ids:
#             relative_path = img_names[i]
#             # 1. 尝试直接拼接 (例如 ./data/Pig/fold1/img.png)
#             full_path = os.path.join(self.root, relative_path)
            
#             # 2. 安全检查：如果 root 路径已经包含了数据集名字 (例如 ./data/Pig)，
#             # 而 relative_path 也是 'Pig/...'，这样拼接会变成 './data/Pig/Pig/...' (这是不对的)
#             # 所以我们要检查一下
#             if not os.path.exists(full_path):
#                 # 尝试去掉 root 的最后一层目录再拼接
#                 # 假设 root='./data/Pig', path='Pig/fold1/...' -> 变成 './data/Pig/fold1/...'
#                 parent_root = os.path.dirname(self.root.rstrip('/')) 
#                 alt_path = os.path.join(parent_root, relative_path)
#                 if os.path.exists(alt_path):
#                     full_path = alt_path
            
#             self.data.append(full_path)
#         self.weights = weights[req_ids]
#         self.targets = np.copy(self.clean_labels)
#         majority_label = np.argmax(self.weights,axis=1)
#         clean_majority = sum(majority_label == self.clean_labels)
                
#         print('Average candidate num: ', self.soft_labels.sum(1).mean())
#         print('clean_num', 
#               sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1),'/',len(self.clean_labels))
#         print('clean majority', clean_majority)
#         wandb.log({
#             'total_labels': len(self.clean_labels),
#             'clean_labels': sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1),
#             'clean_majority_labels': clean_majority
#         })
#         # ======================================================
#         # 🚀 在这里添加以下代码来检查均衡性
#         # ======================================================
#         print(f"\n--- 类别均衡性检查 (Class Balance Check) ---")
#         # 1. 使用 NumPy 统计每个类别的数量
#         #    np.unique 会返回唯一的类别 ID 和它们各自的数量
#         class_ids, class_counts = np.unique(self.clean_labels, return_counts=True)
        
#         # 2. (可选) 创建一个字典或 Pandas DataFrame 来清晰展示
#         print(f"总类别数量: {len(class_ids)}")
#         print("各个类别的样本数分布:")
        
#         # 3. 打印出每个类别和它的数量
#         for cid, count in zip(class_ids, class_counts):
#             print(f"  类别 {cid}: {count} 个样本")
        
#         # 4. 打印一个总结
#         min_count = np.min(class_counts)
#         max_count = np.max(class_counts)
#         print(f"  [总结] 样本数范围: 从 {min_count} (最少) 到 {max_count} (最多)")
#         if max_count / min_count > 5.0: # (你可以自己定义这个阈值)
#             print(f"  [结论] ⚠️ 数据集非常不均衡! (Imbalanced)")
#         elif max_count / min_count > 2.0:
#             print(f"  [结论] ❗ 数据集轻微不均衡 (Mildly Imbalanced)")
#         else:
#             print(f"  [结论] ✅ 数据集相对均衡 (Relatively Balanced)")
#         print("--------------------------------------------------\n")
#         # ======================================================
#         # 结束
#         # ======================================================
#     def partial_noise(self):
#         self.targets = np.zeros((len(self.clean_labels),))-1

#     def __getitem__(self, index):

#         img, labels = self.data[index], self.targets[index]
#         # img = Image.open(img)
#         img = Image.open(img).convert('RGB')  # <--- 强制转为3通道彩色
#         img = self.transform(img)
#         if self.train:
#             return img, labels, index
#         else:
#             return img, labels
        
#     def __len__(self):
#         return len(self.targets)
import json
import numpy as np
import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset
import wandb

def get_dataset(args, transform_train, transform_test):
    # 根据 Slice 定义 Fold
    if args.slice == 1:
        train_split, test_split = ['fold1', 'fold2', 'fold4', 'fold5'], ['fold3']
    elif args.slice == 2:
        train_split, test_split = ['fold1','fold2','fold5'],['fold4']
    elif args.slice == 3:
        train_split, test_split = ['fold1','fold2','fold3'],['fold5']
    
    # 实例化 Dataset
    trainset = Crowdsource(args, splits=train_split, transform=transform_train)
    trainset.partial_noise() # 将 targets 设为 -1 (未标记状态)
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

        # 1. 更加健壮的路径读取
        annotation_file = os.path.join(self.root, 'annotations.json')
        if not os.path.exists(annotation_file):
            # 尝试在子文件夹里找 (兼容 ./data/DatasetName 结构)
            # 假设 args.train_root 只是 ./data，但我们需要去 ./data/DatasetName/annotations.json
            potential_path = os.path.join(self.root, args.dataset, 'annotations.json')
            if os.path.exists(potential_path):
                self.root = os.path.join(self.root, args.dataset)
                annotation_file = potential_path
        
        print(f"Loading annotations from: {annotation_file}")
        with open(annotation_file, 'r') as outfile:
            annotation_jsons = json.load(outfile)
            
        # 兼容不同的 JSON 结构 (list vs dict)
        if isinstance(annotation_jsons, list):
            raw_entries = annotation_jsons[0]["annotations"]
        else:
            raw_entries = annotation_jsons["annotations"]

        # 2. 扫描所有图片和标签 (建立 String -> Int 映射)
        img_names_set = set()
        unique_labels_set = set()
        
        for entry in raw_entries:
            if entry["class_label"] is not None:
                img_names_set.add(entry["image_path"])
                unique_labels_set.add(entry["class_label"])
                
        img_names = sorted(list(img_names_set))
        
        # 尝试智能排序标签 (如果是数字字符串 '1', '2', '10'，按数字排；否则按字母排)
        try:
            unique_labels = sorted(list(unique_labels_set), key=lambda x: int(x))
        except:
            unique_labels = sorted(list(unique_labels_set))
            
        print(f"Found {len(unique_labels)} unique labels: {unique_labels[:5]}...")
        
        # --- 关键修复: 建立映射字典 ---
        # label_to_idx: {'head_injury': 0, 'not_injured': 1, ...}
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        # 警告：如果 args.num_classes 设置得不对
        if len(unique_labels) != self.num_classes:
            print(f"Warning: args.num_classes={self.num_classes} but found {len(unique_labels)} unique labels in JSON.")
            print(f"Please update your --num_classes_map in the training script to {len(unique_labels)}!")

        # 建立图片名映射
        map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
        
        # 3. 处理 Fold
        self.folds = {}
        for name in img_names:
            # 兼容 Windows/Linux 路径分隔符
            fold = name.split('/')[1] if '/' in name else name.split('\\')[1]
            if fold in self.folds:
                self.folds[fold].append(map_names[name])
            else:
                self.folds[fold] = [map_names[name]]
        
        # 4. 构建投票矩阵 _data
        # 强制使用 self.num_classes 作为列数，防止维度不匹配
        _data = np.zeros((len(img_names), self.num_classes))

        for entry in raw_entries:
            if entry["class_label"] is not None:
                img_idx = map_names[entry["image_path"]]
                label_str = entry["class_label"]
                
                # 使用字典映射将 String 转为 Int
                if label_str in self.label_to_idx:
                    label_idx = self.label_to_idx[label_str]
                    if label_idx < self.num_classes:
                        _data[img_idx, label_idx] += 1
                    else:
                        pass # 忽略超出范围的标签 (如果 num_classes 设小了)

        # 5. LPI 采样 (核心逻辑)
        data = np.zeros((len(img_names), self.num_classes))
        rng = np.random.default_rng(self.args.seed_dataset)
        
        for i in range(data.shape[0]):
            total_votes = _data[i].sum()
            if total_votes > 0:
                probs = _data[i] / total_votes
                # 这里 a=num_classes, p=probs (长度也是 num_classes)，完美匹配！
                annots = rng.choice(self.num_classes, p=probs, size=args.lpi)
                for a in annots:
                    data[i, a] += 1
            else:
                # 极少数情况图片无标注，随机分配一个以防报错
                pass
                
        partialY = np.zeros((len(img_names), self.num_classes))
        partialY[data.nonzero()] = 1
        
        weights = data / (data.sum(1, keepdims=True) + 1e-8)
        clean_labels = np.argmax(_data, axis=1)
        
        # 6. 筛选 Split 数据
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
        
        # --- 关键修复: 生成绝对路径 ---
        self.data = []
        for i in req_ids:
            relative_path = img_names[i]
            full_path = os.path.join(self.root, relative_path)
            
            # 路径回退检查 (防止 ./data/Pig/Pig/...)
            if not os.path.exists(full_path):
                # 尝试去掉最后一层 (假设 root 是 .../Pig 而图片路径也是 Pig/...)
                parent_root = os.path.dirname(self.root.rstrip(os.sep))
                alt_path = os.path.join(parent_root, relative_path)
                if os.path.exists(alt_path):
                    full_path = alt_path
            
            self.data.append(full_path)

        # 7. 打印统计信息
        majority_label = np.argmax(self.weights, axis=1)
        clean_majority = sum(majority_label == self.clean_labels)
        
        print('Average candidate num: ', self.soft_labels.sum(1).mean())
        print('clean_majority', clean_majority)
        
        # 类别均衡性检查
        print(f"\n--- Class Balance Check ({args.dataset}) ---")
        class_ids, class_counts = np.unique(self.clean_labels, return_counts=True)
        print("Distribution:")
        for cid, count in zip(class_ids, class_counts):
            # 尝试把 ID 转换回 Label 名字显示，更直观
            label_name = [k for k, v in self.label_to_idx.items() if v == cid]
            label_name = label_name[0] if label_name else str(cid)
            print(f"  Class {cid} ({label_name}): {count}")
        print("---------------------------------------------\n")

    def partial_noise(self):
        self.targets = np.zeros((len(self.clean_labels),)) - 1

    def __getitem__(self, index):
        img_path, labels = self.data[index], self.targets[index]
        
        # --- 关键修复: 强制 RGB ---
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            # 返回全黑图防止训练崩溃
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        if self.train:
            return img, labels, index
        else:
            return img, labels
        
    def __len__(self):
        return len(self.targets)    