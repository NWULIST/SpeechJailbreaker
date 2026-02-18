import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchaudio
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class AudioInspector:
    """
    Utility class to register a forward hook and collect outputs from a target layer.

    Parameters
    ----------
    targetLayer : torch.nn.Module
        The layer to inspect.
    """
    def __init__(self, targetLayer):
        self.layerOutputs = []
        self.featureHandle = targetLayer.register_forward_hook(self.feature)
    
    def feature(self, model, input, output):
        self.layerOutputs.append(output.detach().cpu())
    
    def release(self):
        """Remove the registered forward hook."""
        self.featureHandle.remove()

def get_activations(model, audio_input: torch.Tensor, activations = 'lm') -> torch.Tensor:
    """
    Get activations from specified layers in QWEN2-Audio model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.
    audio_input : torch.Tensor
        Input tensor for the model.
    activations : str, optional
        Which activations to extract ('lm' or 'audio'). Default is 'lm'.

    Returns
    -------
    torch.Tensor
        Activations tensor of shape (seq_len, num_layers, hidden_size).
    """
    model.eval()
    with torch.no_grad():
        # Only create inspectors for the requested activation type
        if activations == 'lm':
            inspectors = [AudioInspector(layer.mlp.act_fn) 
                         for layer in model.language_model.model.layers]
        elif activations == 'audio':
            inspectors = [AudioInspector(layer.activation_fn) 
                         for layer in model.audio_tower.layers]
        else:
            raise ValueError(f"Unknown activation type: {activations}")
        
        # Run model forward pass
        outputs = model(**audio_input)
        
        # Process activations and cleanup immediately
        try:
            activations_tensor = torch.cat([torch.cat(inspector.layerOutputs, dim=1) 
                                          for inspector in inspectors], dim=0).transpose(0,1)
        finally:
            # Ensure hooks are released even if concatenation fails
            for inspector in inspectors:
                inspector.release()

        return activations_tensor

def detect_noise_sensitive_neurons(model,
                                   clean_audio: torch.Tensor,
                                   noisy_audio: torch.Tensor,
                                   top_k_percent: float = 0.5,
                                   activations = 'lm') -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
    """
    Detect neurons most sensitive to noise based on activation differences.

    Parameters
    ----------
    model : torch.nn.Module
        The Qwen2-Audio model.
    clean_audio : torch.Tensor
        Clean audio input tensor.
    noisy_audio : torch.Tensor
        Noisy audio input tensor.
    top_k_percent : float, optional
        Percentage of top sensitive neurons to select. Default is 0.5.
    activations : str, optional
        Type of activations to analyze ('lm' or 'audio'). Default is 'lm'.

    Returns
    -------
    noise_sensitive_neurons : List[Tuple[int, int]]
        List of (layer_idx, neuron_idx) pairs for most sensitive neurons.
    clean_acts : torch.Tensor
        Clean activations tensor.
    noisy_acts : torch.Tensor
        Noisy activations tensor.
    """
    # Get activations for clean and noisy inputs
    clean_acts = get_activations(model, clean_audio, activations)
    noisy_acts = get_activations(model, noisy_audio, activations)
    
    # Sum activations over sequence length
    clean_acts_sum = clean_acts.sum(dim=0)
    noisy_acts_sum = noisy_acts.sum(dim=0)
    
    act_diff = (noisy_acts_sum - clean_acts_sum).abs()
    
    # Find top-k% neurons with the highest activation differences
    hidden_size = act_diff.shape[1]
    total_neurons = act_diff.numel()
    k = int(total_neurons * (top_k_percent / 100))
    
    # Flatten and get top-k indices
    act_diff_flat = act_diff.flatten()
    top_k_values, top_k_indices = torch.topk(act_diff_flat, k)
    
    # Convert flat indices to (layer_idx, neuron_idx) pairs
    noise_sensitive_neurons = []
    for idx in top_k_indices:
        layer_idx = (idx // hidden_size).item()
        neuron_idx = (idx % hidden_size).item()
        noise_sensitive_neurons.append((layer_idx, neuron_idx))
    
    return noise_sensitive_neurons, clean_acts.to('cuda'), noisy_acts.to('cuda')

def detect_random_neurons(model,
                          clean_audio: torch.Tensor,
                          noisy_audio: torch.Tensor,
                          top_k_percent: float = 0.5,
                          activations = 'lm') -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
    """
    Select random neurons as a baseline comparison for noise-sensitive neuron detection.

    Parameters
    ----------
    model : torch.nn.Module
        The Qwen2-Audio model.
    clean_audio : torch.Tensor
        Clean audio input tensor.
    noisy_audio : torch.Tensor
        Noisy audio input tensor.
    top_k_percent : float, optional
        Percentage of total neurons to randomly select. Default is 0.5.
    activations : str, optional
        Type of activations to analyze ('lm' or 'audio'). Default is 'lm'.

    Returns
    -------
    random_neurons : List[Tuple[int, int]]
        List of (layer_idx, neuron_idx) pairs for randomly selected neurons.
    clean_acts : torch.Tensor
        Clean activations tensor.
    noisy_acts : torch.Tensor
        Noisy activations tensor.
    """
    # Get activations for clean and noisy inputs
    clean_acts = get_activations(model, clean_audio, activations)
    noisy_acts = get_activations(model, noisy_audio, activations)

    # Ensure shape is (seq_len, num_layers, hidden_size)
    seq_len, num_layers, hidden_size = clean_acts.shape

    # Total neurons across all layers
    total_neurons = num_layers * hidden_size  
    k = int(total_neurons * (top_k_percent / 100))

    # Select k random neurons
    top_k_indices = torch.randperm(total_neurons)[:k]

    # Convert flat indices to (layer_idx, neuron_idx) pairs
    random_neurons = []
    for idx in top_k_indices:
        layer_idx = (idx // hidden_size).item()  # Corrected calculation
        neuron_idx = (idx % hidden_size).item()
        random_neurons.append((layer_idx, neuron_idx))

    return random_neurons, clean_acts.to('cuda'), noisy_acts.to('cuda')


def visualize_neuron_distribution(noise_neurons: List[Tuple[int, int]], 
                                num_layers: int,
                                title: str = "Noise-Sensitive Neuron Distribution"):
    """
    Visualize distribution of noise-sensitive neurons across layers.

    Parameters
    ----------
    noise_neurons : List[Tuple[int, int]]
        List of (layer_idx, neuron_idx) pairs.
    num_layers : int
        Number of layers in the model.
    title : str, optional
        Title for the plot. Default is "Noise-Sensitive Neuron Distribution".
    """
    layer_counts = [0] * num_layers
    for layer_idx, _ in noise_neurons:
        layer_counts[layer_idx] += 1
        
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_layers), layer_counts)
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Number of Noise-Sensitive Neurons")
    plt.show()
