# common/model.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyNN(nn.Module):
    """非常にシンプルな3層ネットワーク (4->4->3->4)"""
    def __init__(self):
        super(TinyNN, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _make_resnet_cifar_stem(model: nn.Module) -> nn.Module:
    # ImageNet stem: 7x7 s2 + maxpool を CIFAR stem: 3x3 s1 + no maxpool にする
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _maybe_make_cifar_stem(model: nn.Module, cifar_stem: bool) -> nn.Module:
    """
    ResNet/ResNeXt/WideResNet は conv1/maxpool を持つので CIFAR stem 化できる。
    VGG/ViT などは対象外。
    """
    if not cifar_stem:
        return model
    if hasattr(model, "conv1") and hasattr(model, "maxpool"):
        return _make_resnet_cifar_stem(model)
    return model


def get_benchmark_model(name: str, num_classes: int = 10, cifar_stem: bool = True):
    name = name.lower()

    # -------------------------
    # ResNet family
    # -------------------------
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet34":
        model = models.resnet34(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet101":
        model = models.resnet101(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet152":
        model = models.resnet152(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -------------------------
    # ResNeXt / WideResNet family
    # -------------------------
    elif name == "resnext101_32x8d":
        model = models.resnext101_32x8d(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # 追加：WideResNet50-2（コスパが良い “強い” 選択肢）
    elif name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "wide_resnet101_2":
        model = models.wide_resnet101_2(weights=None)
        model = _maybe_make_cifar_stem(model, cifar_stem)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -------------------------
    # VGG
    # -------------------------
    elif name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    elif name == "vgg16_bn":
        model = models.vgg16_bn(weights=None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    # -------------------------
    # ViT (CIFARでは resize_to_imagenet=True 前提になりがち)
    # -------------------------
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    elif name == "vit_l_16":
        model = models.vit_l_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    
    # -------------------------
    # DistilBERT (Text)
    # -------------------------
    elif name in ("distilbert", "distilbert-base-uncased"):
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=num_classes,
        )
        return model
    
    elif name in ("squeezebert"):
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "squeezebert/squeezebert-uncased",
            num_labels=num_classes
        )
        return model
    
    elif name in ("deberta"):
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-small",
            num_labels=num_classes
        )
        return model
    
    elif name in ("minilm-l12-h384", "minilm", "microsoft/minilm-l12-h384-uncased"):
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",
            num_labels=num_classes,
        )
        return model
    
    elif name in ("bert-medium"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_id = "nreimers/BERT-Medium_L-8_H-512_A-8"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
        )
        return model

    # -------------------------
    # Causal LM (OPT, LLaMA)
    # -------------------------
    elif name in ("opt-1.3b", "opt_1.3b"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
        model.config.use_cache = False  # 学習時はKVキャッシュ不要（DynamicCache警告回避）
        return model

    elif name in ("llama-3b", "llama_3b", "llama-3.2-3b"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        model.config.use_cache = False
        return model

    elif name in ("llama-7b", "llama_7b", "llama-2-7b"):
        from transformers import AutoModelForCausalLM
        # meta-llama/Llama-2-7b-hf はゲート付き。
        # HuggingFace tokenでログイン + Metaのライセンス承諾が必要。
        # huggingface-cli login で事前にトークンを設定すること。
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        model.config.use_cache = False  # 学習時はKVキャッシュ不要（DynamicCache警告回避）
        return model

    # -------------------------
    # Masked LM (DeBERTa)
    # -------------------------
    elif name in ("deberta-xl", "deberta_xl"):
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-xlarge")
        # DeBERTaのDebertaEncoder.get_rel_embedding()はrel_embeddings.weight
        # (Parameter)をそのまま返すため、ZeRO-3のフックがTensorとして検知できず
        # WARNING が大量に出る。+演算でParameterをTensorに変換して回避する。
        from transformers.models.deberta.modeling_deberta import DebertaEncoder
        for module in model.modules():
            if isinstance(module, DebertaEncoder) and module.relative_attention:
                _orig_fn = module.get_rel_embedding
                def _patched(orig=_orig_fn):
                    emb = orig()
                    return emb + 0 if emb is not None else None
                module.get_rel_embedding = _patched
        return model

    # -------------------------
    # TinyNN (デバッグ用)
    # -------------------------
    elif name == "tinynn":
        return TinyNN()

    else:
        raise ValueError(f"Unknown model name {name}")
