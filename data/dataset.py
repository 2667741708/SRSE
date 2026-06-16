from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings

from torch.cuda.amp import GradScaler
import faiss
warnings.filterwarnings('ignore')

from utils.AverageMeter import *
from utils.other_utils import *
from utils.utils_mixup import *
from utils.losses import *

import wandb
import os
import pickle
import torchvision as tv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
def get_dataset(args, transform_train, transform_test):

    if args.dataset == 'CIFAR-100':
        trainset = CIFAR100Partial(args, transform=transform_train, target_transform=transform_test, download=args.download)
        testset = tv.datasets.CIFAR100(root=args.train_root, train=False, download=args.download, transform=transform_test)
        if args.noise_type == 'partial':
            trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio, heirarchical=args.heirarchical)
        

    elif args.dataset == 'CIFAR-10':
        trainset = CIFAR10Partial(args, transform=transform_train, target_transform=transform_test, download=args.download)
        testset = tv.datasets.CIFAR10(root=args.train_root, train=False, download=args.download, transform=transform_test)
        if args.noise_type == 'partial':
           trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio)

    elif args.dataset == 'CUB-200':
        trainset = CUB200Partial(args, train=True, transform=transform_train)
        testset = CUB200Partial(args, train=False, transform=transform_test)
        if args.noise_type == 'partial':
            trainset.partial_noise(partial_rate=args.partial_ratio, noisy_rate=args.noise_ratio)
       
    return trainset, testset


class CUB200Partial(Dataset):
    """
    一个现代化的 CUB200Partial 类，它通过文件路径加载数据，
    而不是依赖预处理的 .pkl 文件。
    """
    base_folder = 'CUB_200_2011'
    
    def __init__(self, args, train=True, transform=None):
        self.root = os.path.join(os.path.expanduser(args.train_root), 'cub200') # <-- 关键路径修正
        self.transform = transform
        self.train = train
        self.args = args
        self.num_classes = 200 # CUB-200 有200个类别

        # 检查数据完整性并加载元数据
        self._load_metadata()

        # 根据 train 参数分割数据
        split_mask = self.data['is_training_img'] == (1 if self.train else 0)
        self.data = self.data[split_mask]

        # 现在 self.data 是一个 pandas DataFrame
        # 为了与你的其他代码兼容，我们创建 data_paths 和 targets 属性
        self.data_paths = self.data['filepath'].tolist()
        self.targets = self.data['target'].tolist() # targets 已经是 0-indexed 了

        # 初始化 soft_labels 和 clean_labels
        self.soft_labels = None
        self.clean_labels = np.array(self.targets)
        
        print(f"CUB200 {'Train' if self.train else 'Test'} set loaded. Samples: {len(self.data_paths)}")

    def _load_metadata(self):
        images_path = os.path.join(self.root, self.base_folder, 'images.txt')
        labels_path = os.path.join(self.root, self.base_folder, 'image_class_labels.txt')
        split_path = os.path.join(self.root, self.base_folder, 'train_test_split.txt')

        if not all(os.path.exists(p) for p in [images_path, labels_path, split_path]):
             raise RuntimeError(f"CUB-200 metadata not found. Please ensure '{self.base_folder}' is extracted in '{self.root}'. You might need to run preprocess_cub200.py first.")

        images = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(labels_path, sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.data.target = self.data.target - 1  # 将标签从 1-200 转换为 0-199

    def partial_noise(self, partial_rate, noisy_rate):
        if not self.train:
            return # 只对训练集加噪声
            
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.array(self.targets)
        
        # 使用你已有的噪声生成函数
        partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)
        self.soft_labels = np.asarray(partialY)

        print('CUB-200 partial noise generated.')
        print('Average candidate num:', self.soft_labels.sum(1).mean())
        print('Clean label in candidate set num:', sum(self.soft_labels[range(len(self.clean_labels)), self.clean_labels] == 1))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.base_folder, 'images', self.data_paths[index])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        # 保持与你其他代码一致的返回格式
        if self.train:
            # 训练时，targets 通常是-1，而真实标签在 soft_labels 或 clean_labels 中
            # 但你的主循环似乎不需要从这里返回标签，所以我们返回index
            return img, -1, index
        else:
            # 测试时，返回图像和真实标签
            return img, self.targets[index]

    def __len__(self):
        return len(self.data_paths)

class CIFAR10Partial(tv.datasets.CIFAR10):

    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(CIFAR10Partial, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])
        
        self.num_classes = self.args.num_classes

    def partial_noise(self, partial_rate, noisy_rate):
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.copy(self.targets)
        clean_labels = np.copy(self.targets)


        partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)

        self.soft_labels = np.asarray(partialY)
        
        temp = torch.zeros(partialY.shape)
        temp[torch.arange(partialY.shape[0]), clean_labels] = 1
        if torch.sum(partialY * temp) == partialY.shape[0]:
            print('partialY correctly loaded')
        else:
            print('inconsistent permutation')
        print('Average candidate num: ', partialY.sum(1).mean())

        self.targets = np.zeros((len(self.targets),))-1
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1))
        #print('clean_num',sum(self.targets==self.clean_labels))


    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels




class CIFAR100Partial(tv.datasets.CIFAR100):

    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(CIFAR100Partial, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])
        
        self.num_classes = self.args.num_classes
        self.clean_labels = np.copy(self.targets)

    def partial_noise(self, partial_rate, noisy_rate, heirarchical):
        np.random.seed(self.args.seed_dataset)
        self.clean_labels = np.copy(self.targets)
        clean_labels = np.copy(self.targets)

        if heirarchical:
            partialY = generate_hierarchical_cv_candidate_labels('cifar100', self.clean_labels, partial_rate, noisy_rate=noisy_rate)
        else:
            partialY = generate_uniform_cv_candidate_labels(self.clean_labels, partial_rate, noisy_rate=noisy_rate)

        self.soft_labels = np.asarray(partialY)
        
        temp = torch.zeros(partialY.shape)
        temp[torch.arange(partialY.shape[0]), clean_labels] = 1
        if torch.sum(partialY * temp) == partialY.shape[0]:
            print('partialY correctly loaded')
        else:
            print('inconsistent permutation')
        print('Average candidate num: ', partialY.sum(1).mean())

        self.targets = np.zeros((len(self.targets),))-1
        print('clean_num', 
              sum(self.soft_labels[range(len(self.clean_labels)),self.clean_labels] == 1))



    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels



def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res



def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1, noisy_rate=0):
    train_labels = torch.tensor(train_labels)
    assert dataname == 'cifar100'

    # meta = unpickle('dataset/cifar-100-python/meta')
    meta = unpickle('data/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
    fish#aquarium fish, flatfish, ray, shark, trout
    flowers#orchid, poppy, rose, sunflower, tulip
    food containers#bottle, bowl, can, cup, plate
    fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
    household electrical devices#clock, keyboard, lamp, telephone, television
    household furniture#bed, chair, couch, table, wardrobe
    insects#bee, beetle, butterfly, caterpillar, cockroach
    large carnivores#bear, leopard, lion, tiger, wolf
    large man-made outdoor things#bridge, castle, house, road, skyscraper
    large natural outdoor scenes#cloud, forest, mountain, plain, sea
    large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals#fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates#crab, lobster, snail, spider, worm
    people#baby, boy, girl, man, woman
    reptiles#crocodile, dinosaur, lizard, snake, turtle
    small mammals#hamster, mouse, rabbit, shrew, squirrel
    trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    vehicles 1#bicycle, bus, motorcycle, pickup truck, train
    vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)
    
    print("Finish Generating Heirarchical Candidate Label Sets!\n")
    return partialY



def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1, noisy_rate=0):
    
    train_labels = torch.tensor(train_labels)
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)

    if noisy_rate == 0:
        partialY[torch.arange(n), train_labels] = 1.0
        # if supervised, reset the true label to be one.
        print('Reset true labels')

    print("Finish Generating Candidate Label Sets!\n")
    return partialY



def train_algo(args, scheduler, model,  device, 
              train_loader, train_selected_loader, optimizer, epoch):
    
    train_loss = AverageMeter()

    model.train()
    end = time.time()
    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
    scaler = GradScaler()

    for batch_idx, (img, labels, index) in enumerate(train_selected_loader):

        model.zero_grad()
        img1, img2, labels = img[0].to(device), img[1].to(device), labels.to(device)

        if args.mixup:
            img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
            img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds1, _ = model(img1)
                preds2, _ = model(img2)

                if args.cr:
                    loss = ClassificationLoss(args, preds1, preds2, y_a1, y_b1, y_a2, y_b2,
                                        lam1, lam2, criterionCE, epoch, device)
                else:
                    loss = ClassificationLoss2(args, preds2, y_a2, y_b2, 
                                               lam2, criterionCE, epoch, device)          
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds1, _ = model(img1)
                preds2, _ = model(img2)

            if args.cr:
                loss = ClassificationLoss4(args, preds1, preds2, labels, criterionCE, epoch, device)
            else:
                loss = ClassficationLoss3(args, preds2, labels, criterionCE, epoch, device)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss.update(loss.item(), img1.size(0))        
          
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    print('train_class_loss',train_loss.avg)
    print('train time', time.time()-end)



def reliable_pseudolabel_selection(args, device, trainloader, features, epoch, model_preds=None):
    
    features_numpy = features.cpu().numpy() 
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    index.add(features_numpy)
    partial_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    clean_labels = torch.LongTensor(trainloader.dataset.clean_labels)
    soft_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))

    D,I = index.search(features_numpy, args.k_val+1)
    neighbors = torch.LongTensor(I)
    weights = torch.exp(torch.Tensor(D[:,0:])/0.1)
    N = features_numpy.shape[0]

    if epoch > args.start_correct:
        prob, pred = torch.max(model_preds,1)
        prob, pred = prob.squeeze(), pred.squeeze()      
        
        conf_th = args.conf_th_h - (args.conf_th_h - args.conf_th_l) * ((epoch - args.start_correct)/(args.epoch - args.start_correct))
        conf_id = (prob > conf_th).nonzero().reshape(-1)
        print('Confident model predictions:', len(conf_id))
        print('Correct confident model predictions:',(pred[conf_id] == clean_labels[conf_id]).sum())
        print('Model pred already in partial set:',(partial_labels[conf_id, pred[conf_id]] == 1).sum())

        soft_labels[conf_id, pred[conf_id]] = 1
        labels[conf_id, pred[conf_id]] = 1


    print('New Clean pl:',(labels[range(N),clean_labels] == 1).sum())
    wandb.log({'Clean pl':(labels[range(N),clean_labels] == 1).sum()},epoch)

    score = torch.zeros(N, args.num_classes)

    knn_indices = neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    )

    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = torch.zeros((len(pseudo_labels),args.num_classes)).scatter_(1, pseudo_labels.view(-1,1), 1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo labels:', correct_soft_labels)
    print('Correct pseudo label matches:', correct_soft_label_matches, match_id.int().sum())


    wandb.log({
        'Correct pseudo labels': correct_soft_labels,
        'Correct pseudo label matches': correct_soft_label_matches,
        'Total noisy_pseudo matches': match_id.int().sum()
    }, epoch)


    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct pseudo label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    ) 


    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = score/score.sum(1).unsqueeze(-1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()


    # correct_soft_labels = ((torch.max(soft_labels, dim=1)[1])[clean_id] == clean_labels[clean_id]).sum()
    # match_id = (torch.max(soft_labels, dim=1)[1] == labels) # only one for every image
    # correct_soft_label_matches = (labels[match_id] == clean_labels[match_id]).int().sum()

    print('Correct posterior labels:', correct_soft_labels)
    print('Correct posterior label matches:', correct_soft_label_matches)

    wandb.log({
        'Correct posterior labels': correct_soft_labels,
        'Correct posterior label matches': correct_soft_label_matches
    }, epoch)

    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct posterior label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct posterior label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    prob_temp = soft_labels[:,:].clone()
    prob_temp[prob_temp<=1e-2] = 1e-2
    prob_temp[prob_temp>(1-1e-2)] = 1-1e-2
    discrepancy_measure2 = -torch.log(prob_temp)

    agreement_measure = torch.zeros((N, args.num_classes)).float()
    agreement_measure[range(N),torch.max(soft_labels, dim=1)[1]] = (labels[range(N),torch.max(soft_labels, dim=1)[1]] == 1.0).float().data.cpu()
        
    print('Init Matches 2:', (agreement_measure[range(N), clean_labels] == 1.0).int().sum())

    num_clean_per_class = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        num_clean_per_class[i] = torch.sum(agreement_measure[:,i])

    if(args.delta==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.delta==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.delta==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.delta)

    print(num_clean_per_class)
    print(num_samples2select_class)

    agreement_measure = torch.zeros((len(labels),))
    selected_examples_labels = torch.zeros((len(clean_labels),args.num_classes))+float('inf')


    for i in range(args.num_classes):
        idx_class = labels[:,i] == 1.0
        samples_per_class = idx_class.sum()
        idx_class = (idx_class.float()==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class, i]

        k_corrected = min(num_samples2select_class, samples_per_class)
        val, top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
        selected_examples_labels[idx_class[top_clean_class_relative_idx],i] = val

    _,selected_labels = torch.min(selected_examples_labels,1)

    selected_examples = agreement_measure
    print('selected examples', sum(selected_examples))

    correct_selected_examples = (selected_labels[selected_examples.bool()] == clean_labels[selected_examples.bool()]).int().sum()
    print('Correct Selected examples:',correct_selected_examples)

    wandb.log({
        'Selected examples': sum(selected_examples),
        'Correct selected examples': correct_selected_examples,
    }, epoch)

    return selected_examples, selected_labels




def reliable_pseudolabel_selection_weighted(args, device, trainloader, features, epoch, model_preds=None):
    
    features_numpy = features.cpu().numpy() 
    index = faiss.IndexFlatIP(features_numpy.shape[1])
    index.add(features_numpy)
    partial_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    clean_labels = torch.LongTensor(trainloader.dataset.clean_labels)
    soft_labels = torch.Tensor(np.copy(trainloader.dataset.soft_labels))
    prior = torch.Tensor(np.copy(trainloader.dataset.weights))


    D,I = index.search(features_numpy, args.k_val+1)
    neighbors = torch.LongTensor(I)
    weights = torch.exp(torch.Tensor(D[:,0:])/0.1)
    N = features_numpy.shape[0]

    if epoch > args.start_correct:
        prob, pred = torch.max(model_preds,1)
        prob, pred = prob.squeeze(), pred.squeeze()      
        
        conf_th = args.conf_th_h - (args.conf_th_h - args.conf_th_l) * ((epoch - args.start_correct)/(args.epoch - args.start_correct))
        conf_id = (prob > conf_th).nonzero().reshape(-1)
        print('Confident model predictions:', len(conf_id))
        print('Correct confident model predictions:',(pred[conf_id] == clean_labels[conf_id]).sum())
        print('Model pred already in partial set:',(partial_labels[conf_id, pred[conf_id]] == 1).sum())

        soft_labels[conf_id, pred[conf_id]] = 1
        labels[conf_id, pred[conf_id]] = 1

    soft_labels_p = torch.mul(soft_labels, prior)

    print('New Clean pl:',(labels[range(N),clean_labels] == 1).sum())
    wandb.log({'Clean pl':(labels[range(N),clean_labels] == 1).sum()},epoch)

    score = torch.zeros(N, args.num_classes)

    knn_indices = neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
    knn_soft_labels = soft_labels_p.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    )

    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = torch.zeros((len(pseudo_labels),args.num_classes)).scatter_(1, pseudo_labels.view(-1,1), 1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo labels:', correct_soft_labels)
    print('Correct pseudo label matches:', correct_soft_label_matches, match_id.int().sum())


    wandb.log({
        'Correct pseudo labels': correct_soft_labels,
        'Correct pseudo label matches': correct_soft_label_matches,
        'Total noisy_pseudo matches': match_id.int().sum()
    }, epoch)


    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct pseudo label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct pseudo label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    knn_soft_labels = soft_labels.expand(N, -1, -1)

    score = torch.sum(
        torch.mul(
            torch.gather(knn_soft_labels, 1, knn_indices),
            weights.view(N, -1, 1),
        ),  # batch_size x k x feature_dim
        1,
    ) 


    pseudo_labels = torch.max(score, -1)[1]
    soft_labels = score/score.sum(1).unsqueeze(-1)

    correct_soft_labels = (pseudo_labels == clean_labels).sum()
    match_id = (partial_labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()


    # correct_soft_labels = ((torch.max(soft_labels, dim=1)[1])[clean_id] == clean_labels[clean_id]).sum()
    # match_id = (torch.max(soft_labels, dim=1)[1] == labels) # only one for every image
    # correct_soft_label_matches = (labels[match_id] == clean_labels[match_id]).int().sum()

    print('Correct posterior labels:', correct_soft_labels)
    print('Correct posterior label matches:', correct_soft_label_matches)

    wandb.log({
        'Correct posterior labels': correct_soft_labels,
        'Correct posterior label matches': correct_soft_label_matches
    }, epoch)

    match_id = (labels[range(N),pseudo_labels] == 1) # only one for every image
    correct_soft_label_matches = (pseudo_labels[match_id] == clean_labels[match_id]).sum()

    print('Correct posterior label matches with  pred+partial:', correct_soft_label_matches, match_id.int().sum())

    wandb.log({
        'Correct posterior label matches with  pred+partial': correct_soft_label_matches,
    }, epoch)


    prob_temp = soft_labels[:,:].clone()
    prob_temp[prob_temp<=1e-2] = 1e-2
    prob_temp[prob_temp>(1-1e-2)] = 1-1e-2
    discrepancy_measure2 = -torch.log(prob_temp)

    agreement_measure = torch.zeros((N, args.num_classes)).float()
    agreement_measure[range(N),torch.max(soft_labels, dim=1)[1]] = (labels[range(N),torch.max(soft_labels, dim=1)[1]] == 1.0).float().data.cpu()
        
    print('Init Matches 2:', (agreement_measure[range(N), clean_labels] == 1.0).int().sum())

    num_clean_per_class = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        num_clean_per_class[i] = torch.sum(agreement_measure[:,i])

    if(args.delta==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.delta==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.delta==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.delta)

    print(num_clean_per_class)
    print(num_samples2select_class)

    agreement_measure = torch.zeros((len(labels),))
    selected_examples_labels = torch.zeros((len(clean_labels),args.num_classes))+float('inf')


    for i in range(args.num_classes):
        idx_class = labels[:,i] == 1.0
        samples_per_class = idx_class.sum()
        idx_class = (idx_class.float()==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class, i]

        k_corrected = min(num_samples2select_class, samples_per_class)
        val, top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
        selected_examples_labels[idx_class[top_clean_class_relative_idx],i] = val

    _,selected_labels = torch.min(selected_examples_labels,1)

    selected_examples = agreement_measure
    print('selected examples', sum(selected_examples))

    correct_selected_examples = (selected_labels[selected_examples.bool()] == clean_labels[selected_examples.bool()]).int().sum()
    print('Correct Selected examples:',correct_selected_examples)

    wandb.log({
        'Selected examples': sum(selected_examples),
        'Correct selected examples': correct_selected_examples,
    }, epoch)

    return selected_examples, selected_labels