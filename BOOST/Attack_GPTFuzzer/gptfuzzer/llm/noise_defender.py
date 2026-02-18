import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from types import MethodType
import tqdm

from BOOST.Attack_GPTFuzzer.gptfuzzer.llm.noise_analyzer import detect_random_neurons, detect_noise_sensitive_neurons
import noisereduce as nr
import soundfile as sf

def convert_neurons_to_dict(neurons: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Convert a list of (layer_idx, neuron_idx) tuples to a dictionary mapping layer indices to lists of neuron indices.

    Parameters
    ----------
    neurons : List[Tuple[int, int]]
        List of (layer_idx, neuron_idx) tuples.

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping layer indices to lists of neuron indices.
    """
    layer2neurons = {}
    for layer_idx, neuron_idx in neurons:
        if layer_idx not in layer2neurons:
            layer2neurons[layer_idx] = []
        layer2neurons[layer_idx].append(neuron_idx)
    return layer2neurons

class InterventionManager:
    """
    Manages interventions (pruning, bias, patching) on model activations for specified neurons.

    Parameters
    ----------
    model : torch.nn.Module
        The model to intervene on.
    neurons : list
        List of (layer_idx, neuron_idx) tuples specifying target neurons.
    intervention_type : str, optional
        Type of intervention ('prune', 'bias', 'patch'). Default is 'prune'.
    clean_activations : torch.Tensor, optional
        Clean activations for patching. Default is None.
    mode : str, optional
        Mode of intervention ('all', 'last', 'lastN'). Default is 'all'.
    bias : float, optional
        Bias value to add if intervention_type is 'bias'. Default is 1.0.
    lastN : int, optional
        Number of last tokens to intervene on if mode is 'lastN'. Default is 10.
    activation_choice : str, optional
        Which activations to intervene on ('lm' or 'audio'). Default is 'lm'.
    """
    def __init__(self, model, neurons, intervention_type='prune', clean_activations=None, 
                 mode='all', bias=1.0, lastN=10, activation_choice='lm'):
        self.model = model
        self.layer2neurons = convert_neurons_to_dict(neurons)
        self.intervention_type = intervention_type
        self.clean_activations = clean_activations
        self.mode = mode
        self.bias = bias
        self.lastN = lastN
        self.activation_choice = activation_choice
        self.original_forwards = {}

    def modify_activations(self, activation, layer_idx):
        """
        Apply interventions to all neurons in a layer at once.

        Parameters
        ----------
        activation : torch.Tensor
            Activation tensor to modify.
        layer_idx : int
            Index of the layer to modify.

        Returns
        -------
        torch.Tensor
            Modified activation tensor.
        """
        if activation.shape[1] == 1:  # Skip if single token
            return activation
        neuron_indices = self.layer2neurons[layer_idx]
        if self.mode == 'last':
            if self.intervention_type == 'prune':
                activation[0, -1, neuron_indices] = 0
            elif self.intervention_type == 'bias':
                activation[0, -1, neuron_indices] += self.bias
            elif self.intervention_type == 'patch' and self.clean_activations is not None:
                activation[0, -1, neuron_indices] = self.clean_activations[-1, layer_idx, neuron_indices]
        elif self.mode == 'all':
            if self.intervention_type == 'prune':
                activation[0, :, neuron_indices] = 0
            elif self.intervention_type == 'bias':
                activation[0, :, neuron_indices] += self.bias
            elif self.intervention_type == 'patch' and self.clean_activations is not None:
                activation[0, :, neuron_indices] = self.clean_activations[:, layer_idx, neuron_indices]
        elif self.mode == 'lastN':
            if self.intervention_type == 'prune':
                activation[0, -self.lastN:, neuron_indices] = 0
            elif self.intervention_type == 'bias':
                activation[0, -self.lastN:, neuron_indices] += self.bias
            elif self.intervention_type == 'patch' and self.clean_activations is not None:
                activation[0, -self.lastN:, neuron_indices] = self.clean_activations[-self.lastN:, layer_idx, neuron_indices]
        return activation

    def factory(self, layer_idx):
        """
        Factory for creating new forward methods for intervention.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to modify.

        Returns
        -------
        function
            New forward method for the layer.
        """
        intervention_manager = self
        original_forward = intervention_manager.original_forwards[layer_idx]
        def new_forward_lm(self, x):
            activation = self.act_fn(self.gate_proj(x))
            neuron_indices = intervention_manager.layer2neurons[layer_idx]
            activation = intervention_manager.modify_activations(activation, layer_idx)
            return self.down_proj(activation * self.up_proj(x))
        def new_forward_audio(self, x):
            activation = original_forward(x)
            neuron_indices = intervention_manager.layer2neurons[layer_idx]
            activation = intervention_manager.modify_activations(activation, layer_idx)
            return activation
        if intervention_manager.activation_choice == 'audio':
            return new_forward_audio
        else:
            return new_forward_lm

    def apply(self):
        """
        Apply interventions by replacing forward methods for target layers.
        """
        for layer_idx in self.layer2neurons.keys():
            if self.activation_choice == 'audio':
                target_layer = self.model.audio_tower.layers[layer_idx].activation_fn
            else:  # lm
                target_layer = self.model.language_model.model.layers[layer_idx].mlp
            self.original_forwards[layer_idx] = target_layer.forward
            target_layer.forward = MethodType(self.factory(layer_idx), target_layer)

    def restore(self):
        """
        Restore original forward methods for all modified layers.
        """
        for layer_idx, original_forward in self.original_forwards.items():
            if self.activation_choice == 'audio':
                target_layer = self.model.audio_tower.layers[layer_idx].activation_fn
            else:  # lm
                target_layer = self.model.language_model.model.layers[layer_idx].mlp
            target_layer.forward = original_forward

def run_with_intervention(model, input_data, neurons, intervention_type='prune', 
                        clean_activations=None, activations_choice='lm', 
                        mode='all', bias=1, lastN=10):
    """
    Run model with specified intervention using modified forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        The model to intervene on.
    input_data : dict
        Model input data.
    neurons : list
        List of (layer_idx, neuron_idx) tuples specifying target neurons.
    intervention_type : str, optional
        Type of intervention ('prune', 'bias', 'patch'). Default is 'prune'.
    clean_activations : torch.Tensor, optional
        Clean activations for patching. Default is None.
    activations_choice : str, optional
        Which activations to intervene on ('lm' or 'audio'). Default is 'lm'.
    mode : str, optional
        Mode of intervention ('all', 'last', 'lastN'). Default is 'all'.
    bias : float, optional
        Bias value to add if intervention_type is 'bias'. Default is 1.
    lastN : int, optional
        Number of last tokens to intervene on if mode is 'lastN'. Default is 10.

    Returns
    -------
    output : Any
        Model output after intervention.
    """
    intervention_manager = InterventionManager(
        model=model,
        neurons=neurons,
        intervention_type=intervention_type,
        clean_activations=clean_activations,
        mode=mode,
        bias=bias,
        lastN=lastN,
        activation_choice=activations_choice
    )
    intervention_manager.apply()
    with torch.no_grad():
        output = model.generate(
            **input_data,
            do_sample=False,
            max_new_tokens=40,
            return_dict_in_generate=True,
            output_logits=True
        )
    intervention_manager.restore()
    return output


import os
import json
import time
import random
import argparse
import itertools
from functools import partial

from tqdm import tqdm

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(features, processor):
    text_list = []
    for _ in features:
        query = _['query']
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ]
        text = processor.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            tokenize=False
        )
        text_list.append(text)

    audio_path = [_['audio'] for _ in features]
    data_idxs = [_['data_idx'] for _ in features
                 ]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in features]
    inputs = processor(text=text_list, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    noisy_audio = [_['input_audios'] for _ in features]
    noisy_audio_inputs = processor(text=text_list, audios=noisy_audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    
    clean_audio = [_['denoised_audios'] for _ in features]
    clean_audio_inputs = processor(text=text_list, audios=clean_audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    
    return inputs, audio_path, data_idxs, noisy_audio_inputs, clean_audio_inputs

class AudioTranscriptionDataset(torch.utils.data.Dataset):

    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        data_idx = data['id']
        query = data['query']

        audioarray, sr = sf.read(audio)
        assert sr == processor.feature_extractor.sampling_rate

        input_audios = ffmpeg_read(read_audio(audio), sampling_rate=processor.feature_extractor.sampling_rate)
        denoised_audios = nr.reduce_noise(y=audioarray, sr=processor.feature_extractor.sampling_rate)
        
        query = "<|audio_bos|><|AUDIO|><|audio_eos|>\nListen to the speech and write down its content."

        return {
            'audio': audio,
            'data_idx': data_idx,
            'query': query,

            'input_audios': input_audios,
            'denoised_audios': denoised_audios
        }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = 0
        self._world_size = 1
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


ds_collections = {
    'airbench_level3': {'path': 'airbench_level_3_eval.jsonl'}
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Initialize Whisper model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda').eval()
    model.eval()
    # Initialize Whisper processor
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    random.seed(args.seed)
    dataset = AudioTranscriptionDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    idxs = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, data_idxs, noisy_inputs, clean_inputs) in tqdm(enumerate(data_loader)):
        inputs = inputs.to('cuda')
        noisy_inputs = noisy_inputs.to('cuda')
        clean_inputs = clean_inputs.to('cuda')
        
        with torch.no_grad():
            # output_ids = model.generate(
            #     **inputs,
            #     max_new_tokens=256,
            #     return_dict_in_generate=True,
            #     output_hidden_states=True,
            #     output_scores=True,
            #     use_cache=True
            # )
            neurons, clean_activations, noisy_activations = detect_noise_sensitive_neurons(
                model, 
                noisy_inputs,
                clean_inputs,
                top_k_percent=0.5,
                activations='audio',
            )
            
            output = run_with_intervention(
                model=model,
                input_data=noisy_inputs,
                neurons=neurons,
                intervention_type='bias',
                clean_activations=clean_activations,
                activations_choice='audio',
                mode='all',
                bias=0.5
            )
            output_ids = output.sequences

            output_ids = output_ids[:, inputs.input_ids.size(1):]
        
        # Decode output tokens
        output = processor.batch_decode(output_ids, skip_special_tokens=True)
        
        print(output)
        
        rets.extend(output)
        audio_paths.extend(audio_path)
        idxs.extend(data_idxs)
        
    
    print(f"Evaluating {args.dataset} ...")

    results = []
    for idx, response, audio_path in zip(idxs, rets, audio_paths):
        results.append({
            'idx': idx,
            'response': response,
            'audio_path': audio_path,
        })
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_qwen_asr_bias_0.5_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'))
