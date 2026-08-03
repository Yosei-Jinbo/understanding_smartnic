import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULT_ROOT = "/home/y-jinbo/HasegawaLab/performance_evaluation/common/data"


def _get_targets(dataset):
    """torchvision のバージョンによって .targets or .y の場合があるため吸収"""
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "y"):
        return dataset.y
    raise AttributeError("Dataset has neither .targets nor .y to read labels.")


def _normalize_labels_and_count_classes(labels):
    """任意の整数ラベル配列を 0 始まりに詰めて、クラス数を返す。
    例) Caltech-256 のように 1..257 でも安全に扱える。
    """
    labels = [int(y) for y in labels]
    y_min, y_max = min(labels), max(labels)
    if y_min != 0:
        labels = [y - y_min for y in labels]
    num_classes = max(labels) + 1
    return labels, num_classes


def _split_by_class_indices(labels_zero_based, num_classes, train_per_class=None, train_ratio=None, seed=0):
    """クラスごとにインデックスを集約し、学習/評価へ分割（再現性あり）。
    優先度: train_per_class > train_ratio。どちらも None なら 60/クラス。
    """
    rng = random.Random(seed)
    per_class_indices = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels_zero_based):
        if 0 <= y < num_classes:
            per_class_indices[y].append(idx)

    train_idx, test_idx = [], []
    for idxs in per_class_indices:
        rng.shuffle(idxs)
        if train_per_class is not None:
            k = min(int(train_per_class), len(idxs))
        elif train_ratio is not None:
            k = int(len(idxs) * float(train_ratio))
        else:
            k = min(60, len(idxs))
        train_idx.extend(idxs[:k])
        test_idx.extend(idxs[k:])
    return train_idx, test_idx


def get_dataloaders(
    dataset_name="CIFAR10",
    batch_size=32,
    num_workers=4,
    resize_to_imagenet=False,
    # Caltech-256 用オプション
    caltech256_train_per_class=60,
    caltech256_train_ratio=None,
    caltech256_seed=0,
    caltech256_download=False,  # 手動DL済みを既定想定
    root=DEFAULT_ROOT,
):
    """共通 DataLoader を返す関数（train_loader, test_loader）"""
    train_dataset, test_dataset = get_datasets(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        resize_to_imagenet=resize_to_imagenet,
        caltech256_train_per_class=caltech256_train_per_class,
        caltech256_train_ratio=caltech256_train_ratio,
        caltech256_seed=caltech256_seed,
        caltech256_download=caltech256_download,
        root=root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader


def get_datasets(
    dataset_name="CIFAR10",
    batch_size=32,
    num_workers=4,
    resize_to_imagenet=False,
    cifar_img_size=None,
    # Caltech-256 用
    caltech256_train_per_class=60,
    caltech256_train_ratio=None,
    caltech256_seed=0,
    caltech256_download=False,  # 404回避のため False を既定
    root=DEFAULT_ROOT,
):
    """共通 Dataset を返す関数（train_dataset, test_dataset）"""
    name = dataset_name.lower()

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        if not resize_to_imagenet:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            img_size = 224 if cifar_img_size is None else int(cifar_img_size)
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_dataset  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    elif name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if not resize_to_imagenet:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            img_size = 224 if cifar_img_size is None else int(cifar_img_size)
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_dataset  = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)

    elif name == "food101":
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_dataset = datasets.Food101(root=root, split="train", download=True, transform=train_tf)
        test_dataset  = datasets.Food101(root=root, split="test",  download=True, transform=test_tf)

    elif name in ("places365", "places365_small", "places365-small"):
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_dataset = datasets.Places365(root=root, split="train-standard", small=True, download=True, transform=train_tf)
        test_dataset  = datasets.Places365(root=root, split="val",            small=True, download=True, transform=test_tf)

    elif name == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        img_size = 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        data_root = os.path.join(root, "imagenet")
        train_dir = os.path.join(data_root, "train")
        val_dir   = os.path.join(data_root, "val")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset  = datasets.ImageFolder(val_dir,   transform=test_transform)

    elif name == "caltech256":
        # ---- Caltech-256：まず transform 無しで全体メタを読み、ラベルからクラス数を動的推定
        meta = datasets.Caltech256(root=root, download=caltech256_download, transform=None)
        raw_labels = _get_targets(meta)
        labels_zero, num_classes = _normalize_labels_and_count_classes(raw_labels)

        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_idx, test_idx = _split_by_class_indices(
            labels_zero_based=labels_zero,
            num_classes=num_classes,
            train_per_class=caltech256_train_per_class,
            train_ratio=caltech256_train_ratio,
            seed=caltech256_seed,
        )

        base_train = datasets.Caltech256(root=root, download=False, transform=train_transform)
        base_test  = datasets.Caltech256(root=root, download=False, transform=test_transform)
        train_dataset = Subset(base_train, train_idx)
        test_dataset  = Subset(base_test,  test_idx)

        caltech_dir = os.path.join(root, "caltech256")
        if not os.path.exists(caltech_dir):
            print(f"[WARN] Caltech-256 expected under {caltech_dir}. "
                  f"If download=True fails in your env, please download manually and set caltech256_download=False.")

    elif name == "stl10":
        # STL10: 96x96, 10クラス
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        img_size = 96

        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_dataset = datasets.STL10(
            root=root,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.STL10(
            root=root,
            split="test",
            download=True,
            transform=test_transform,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset