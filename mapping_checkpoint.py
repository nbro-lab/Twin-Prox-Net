"""Strictly load either supported mapping architecture without altering weights."""
import torch
from train_mapping import DeepMTPMapping12Echo
from model_mapping_legacy import DeepMTP_Mapping_12Echo

def load_mapping_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if "learned_S.ImageLayer_S.seq.0.weight" in state:
        model = DeepMTP_Mapping_12Echo()
        architecture = "legacy_pairwise_8_and_10_channels"
    elif "learned_s.spatial_corrector.net.0.weight" in state:
        model = DeepMTPMapping12Echo()
        architecture = "integrated_two_channel"
    else:
        raise ValueError("Unrecognized mapping checkpoint architecture")
    model.load_state_dict(state, strict=True)
    print("Mapping architecture:", architecture)
    return model.to(device).eval(), architecture
