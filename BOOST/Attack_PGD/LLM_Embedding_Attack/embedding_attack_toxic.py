"""
Inspired by the llm-attacks project: https://github.com/llm-attacks/llm-attacks

Run:

    python embedding_attack_toxic.py --help

for more information.
"""

import csv
import torch
import torch.nn as nn
import tqdm
import json
import os
import librosa
from io import BytesIO
from urllib.request import urlopen

from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

def load_qwen2audio_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    """
    加载 Qwen2Audio 模型及对应 tokenizer/processor。
    """

    # 1️⃣ 加载模型
    model = (
        Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 可改为 bfloat16 或 auto
            trust_remote_code=True,
            **kwargs,
        )
        .to(device)
        .eval()
    )

    # 2️⃣ 选择分词器路径
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    # 3️⃣ 加载 processor（Qwen2Audio 用 processor，而不是单纯 tokenizer）
    # processor = AutoProcessor 包含 tokenizer + feature_extractor
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

    tokenizer = processor.tokenizer  # 兼容旧逻辑，方便你和之前函数共用接口

    # 4️⃣ 设置特殊 token 与 padding 规则
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 5️⃣ 返回 model, tokenizer, processor
    return model, tokenizer, processor



def get_embedding_matrix(model):
    """
    获取语言模型的 embedding matrix。

    支持：
    - GPT-J
    - GPT-2
    - LLaMA / LLaMA-2
    - GPT-NeoX
    - Qwen2AudioForConditionalGeneration
    """

    if isinstance(model, Qwen2AudioForConditionalGeneration):
        # Qwen2Audio 模型的文本 embedding
        return model.get_input_embeddings().weight
    elif hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight
    else:
        # Fallback using common attribute names
        try:
            return model.get_input_embeddings().weight
        except:
            raise ValueError(f"❌ Unknown model type: {type(model)}")



def generate(model, input_embeddings, num_tokens=50, input_features=None):
    """
    通用 embedding-level greedy 生成函数。
    支持：
      - GPT/LLAMA 等文本模型
      - Qwen2AudioForConditionalGeneration (可选音频输入)
    """
    model.eval()
    embedding_matrix = get_embedding_matrix(model)[0] # get_embedding_matrix returns weight, we need just weight not [0] if it was tuple.
    # Actually get_embedding_matrix returns a tensor (weight).
    embedding_matrix = get_embedding_matrix(model)

    input_embeddings = input_embeddings.clone()

    generated_tokens = torch.tensor([], dtype=torch.long, device=model.device)

    print("Generating...")
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_tokens)):
            # 对于多模态模型，如 Qwen2Audio，可以传 input_features
            if input_features is not None:
                outputs = model(
                    input_ids=None,
                    inputs_embeds=input_embeddings,
                    input_features=input_features,
                )
            else:
                outputs = model(input_ids=None, inputs_embeds=input_embeddings)

            logits = outputs.logits
            predicted_token = torch.argmax(logits[:, -1, :])

            generated_tokens = torch.cat((generated_tokens, predicted_token.unsqueeze(0)))

            # 获取对应 embedding，拼接到输入序列
            predicted_embedding = embedding_matrix[predicted_token]
            input_embeddings = torch.hstack([input_embeddings, predicted_embedding[None, None, :]])

    return generated_tokens.cpu().numpy()



def calc_loss(
    model,
    embeddings,
    embeddings_attack,
    embeddings_target,
    targets,
    input_features=None,  # 兼容 Qwen2Audio 音频输入
):
    """
    计算拼接后输入的 cross-entropy loss。
    支持文本模型与 Qwen2Audio 等多模态模型。
    """
    # 拼接 embeddings
    full_embeddings = torch.hstack([embeddings, embeddings_attack, embeddings_target])

    # 前向传播
    if input_features is not None:
        outputs = model(inputs_embeds=full_embeddings, input_features=input_features)
    else:
        outputs = model(inputs_embeds=full_embeddings)

    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # 计算 loss 区间
    # embeddings: fixed_prompt
    # embeddings_attack: control_prompt
    # embeddings_target: target
    # We want to predict target given fixed+attack
    
    # logits shape: [batch, seq_len, vocab]
    # The logits we care about start after (len(fixed) + len(attack) - 1)
    # Because logit[i] predicts token[i+1]
    
    len_fixed = embeddings.shape[1]
    len_attack = embeddings_attack.shape[1]
    len_target = embeddings_target.shape[1]
    
    # The last token of attack predicts the first token of target
    # So we start from len_fixed + len_attack - 1
    loss_slice_start = len_fixed + len_attack
    
    # We want logits for [start-1 : end-1]
    # start = loss_slice_start
    # end = loss_slice_start + len_target
    
    # logits for predicting target[0] is at index [loss_slice_start-1]
    # logits for predicting target[last] is at index [loss_slice_start + len_target - 2]
    
    # Slice logits:
    logits_slice = logits[0, loss_slice_start - 1 : loss_slice_start + len_target - 1, :]
    
    # targets should be the token ids of target
    targets_slice = targets
    
    if logits_slice.shape[0] != targets_slice.shape[0]:
        # Truncate to min length if mismatch (should not happen if logic is correct)
        min_len = min(logits_slice.shape[0], targets_slice.shape[0])
        logits_slice = logits_slice[:min_len]
        targets_slice = targets_slice[:min_len]

    loss = nn.CrossEntropyLoss()(logits_slice, targets_slice)

    # Do not return logits to save memory
    return loss


def create_one_hot_and_embeddings(tokens, embed_weights, model):
    one_hot = torch.zeros(
        tokens.shape[0], embed_weights.shape[0], device=model.device, dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        tokens.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    embeddings = (one_hot @ embed_weights).unsqueeze(0).data
    return one_hot, embeddings


class EmbeddingAttack:
    def __init__(self, args):
        self.args = args
        self.num_steps = getattr(args, 'n_steps', 100)
        self.step_size = getattr(args, 'step_size', 0.01)
        self.control_init = getattr(args, 'control_init', "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        # If args.device exists use it, else default to cuda
        self.device = getattr(args, 'device', 'cuda:0')
        self.print_interval = 5
        self.generate_interval = 500
        self.verbose = False
        
    def run(self, model, tokenizer, fixed_prompt, target, audio_path=None, processor=None):
        """
        Run the embedding attack.
        Arguments:
            model: The loaded HuggingFace model.
            tokenizer: The loaded tokenizer.
            fixed_prompt: The question/prompt from the user.
            target: The target response string.
            audio_path: Path or URL to input audio (optional).
            processor: The loaded processor (optional, required if audio_path is provided).
            
        Returns:
            optim_prompts: List of optimized prompts/suffixes (strings).
            steps: List of steps where they were recorded.
            scores: List of loss scores.
        """
        device = model.device
        embed_weights = get_embedding_matrix(model)
        
        control_prompt = self.control_init
        
        # Audio processing
        input_features = None
        if audio_path and processor:
            # Check if using Qwen2Audio or similar that needs input_features
            # Load Audio
            if os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_bytes = BytesIO(f.read())
            else:
                audio_bytes = BytesIO(urlopen(audio_path).read())
            
            # Use processor's feature extractor sampling rate
            sampling_rate = processor.feature_extractor.sampling_rate
            audio_data, _ = librosa.load(audio_bytes, sr=sampling_rate)
            
            # Process audio input features
            inputs = processor(text=fixed_prompt, audios=[audio_data], return_tensors="pt", sampling_rate=sampling_rate)
            if "input_features" in inputs:
                input_features = inputs["input_features"].to(device)
        
        # Tokenize fixed prompt
        input_ids_fixed = tokenizer(fixed_prompt, return_tensors='pt')["input_ids"].to(device)
        if input_ids_fixed.dim() == 2: input_ids_fixed = input_ids_fixed[0]
        
        # Tokenize control prompt
        # Note: Llama tokenizer adds BOS to start by default, correct handling needed
        control_tokens_all = tokenizer(control_prompt, return_tensors='pt')["input_ids"].to(device)
        if control_tokens_all.dim() == 2: control_tokens_all = control_tokens_all[0]
        
        # Tokenize target
        target_tokens_all = tokenizer(target, return_tensors='pt')["input_ids"].to(device)
        if target_tokens_all.dim() == 2: target_tokens_all = target_tokens_all[0]
        
        # Handle BOS token removal for concatenation
        if tokenizer.bos_token_id is not None:
            # removing 'beginning of sentence' token for components that are appended
            if len(control_tokens_all) > 1 and control_tokens_all[0] == tokenizer.bos_token_id:
                control_tokens_all = control_tokens_all[1:]
            if len(target_tokens_all) > 1 and target_tokens_all[0] == tokenizer.bos_token_id:
                target_tokens_all = target_tokens_all[1:]
                
        # Embeddings
        one_hot_inputs, embeddings = create_one_hot_and_embeddings(input_ids_fixed, embed_weights, model)
        one_hot_attack, embeddings_attack = create_one_hot_and_embeddings(control_tokens_all, embed_weights, model)
        one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_tokens_all, embed_weights, model)

        adv_pert = torch.zeros_like(embeddings_attack, requires_grad=True, device=device)
        
        optim_prompts = []
        steps = []
        scores = []

        print(f"Starting Embedding Attack for {self.num_steps} steps...")

        for i in range(self.num_steps):
            adv_pert.requires_grad_()
            
            # Forward pass with perturbation
            current_attack_embeds = embeddings_attack + adv_pert
            
            loss = calc_loss(
                model, embeddings, current_attack_embeds, embeddings_target, target_tokens_all,
                input_features=input_features
            )
            
            loss.backward()
            grad = adv_pert.grad.data
            
            # Update perturbation (Signed Gradient Descent)
            adv_pert.data -= torch.sign(grad) * self.step_size

            model.zero_grad()
            adv_pert.grad.zero_()

            scores.append(loss.item())
            steps.append(i)
            
            # Since we optimize embeddings directly, there is no "text" prompt changing every step.
            # We return a placeholder string or the initial prompt for consistency.
            # If we wanted Soft Prompt Tuning results, we'd save the embeddings or projected tokens.
            # Here we save a descriptive string.
            optim_prompts.append(f"SoftPrompt_Loss_{loss.item():.4f}")
            
            if i % self.print_interval == 0:
                print(f"Iter: {i}, Loss: {loss.item():.4f}")
                
        return optim_prompts, steps, scores

if __name__ == "__main__":
    pass
