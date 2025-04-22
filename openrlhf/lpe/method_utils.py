import torch as th
import blobfile as bf
from tqdm import tqdm
import subprocess
import os

from .utils import distribution_registry

MODEL_NAMES = ["gelu-1l", "gelu-2l", "gelu-4l"]
DISTRIB_NAMES = ["hex", "camel", "indent", "ifelse", "caps", "english", "spanish", "icl"]
N_REPS_DICT = {"hex": 32, "indent": 32, "ifelse": 32, "camel": 32, "caps": 16, "english": 24, "spanish": 24, "icl": 1}
RECOMMENDED_TEMPS = {
    "gelu-1l": {
        "ITGIS": {"hex": 1.0, "camel": 1.0, "indent": 1.0, "ifelse": 1.0, "caps": 1.5, "english": 0.45, "spanish": 0.67, "icl": 0.45},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 2.24, "caps": 3.34, "english": 1.5, "spanish": 2.24, "icl": 1.0}
    },
    "gelu-2l": {
        "ITGIS": {"hex": 1.5, "camel": 1.5, "indent": 1.0, "ifelse": 0.45, "caps": 0.45, "english": 0.67, "spanish": 0.67, "icl": 0.3},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.5, "ifelse": 1.5, "caps": 1.5, "english": 2.24, "spanish": 2.24, "icl": 0.67}
    },
    "gelu-4l": {
        "ITGIS": {"hex": 5.0, "camel": 1.0, "indent": 0.67, "ifelse": 1.0, "caps": 0.67, "english": 0.45, "spanish": 1.0, "icl": 3.34},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 1.0, "caps": 1.0, "english": 1.5, "spanish": 2.24, "icl": 0.67}
    }
}

def load_ground_truth(model_name: str, dist_names: list[str] = DISTRIB_NAMES, device: str = "cpu"):
    """
    Returns a dictionary of tensors of length `vocab_size` denoting the frequencies. The sum of each tensor is 2^32.
    """
    assert model_name in MODEL_NAMES
    gt_freqs = {}
    
    for dist_name in dist_names:
        with bf.BlobFile(f"gs://arc-ml-public/lpe/ground-truth/{model_name}/frequencies32-{dist_name}.pt"
, "rb") as f:
            gt_freqs[dist_name] = th.load(f, map_location=device, weights_only=True)
        
        assert th.sum(gt_freqs[dist_name]).item() == 2**32
    
    return gt_freqs

def gen_activ_samples(model, dist_name: str, n_samples: int, batch_size: int = 64, show_progress: bool = False):
    """
    Generates `n_samples` samples of the pre-unembed activations from the distribution `dist_name`.
    """
    assert dist_name in DISTRIB_NAMES
    dist = distribution_registry[dist_name](model.tokenizer)
    samples = dist.sample(n_reps=N_REPS_DICT[dist_name], n_samples=n_samples)
    acts = []
    with th.no_grad():
        for batch in tqdm(samples.split(batch_size, dim=0)):
            x = model.embed(batch) + model.pos_embed(batch)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x)
            acts.append(x[:,-1,:].clone())
    return th.cat(acts)

def pick_random_tokens(gt: th.Tensor, count, p_min=1e-9, p_max=1e-5):
    """
    Returns `count` random tokens whose ground-truth probabilities are in the range [p_min, p_max].
    """
    gt_probs = gt / gt.sum()
    valid_idx = th.logical_and(gt_probs >= p_min, gt_probs <= p_max).nonzero().squeeze()
    sampled_idx = th.randperm(valid_idx.size(0))[:count]
    return valid_idx[sampled_idx].tolist()
