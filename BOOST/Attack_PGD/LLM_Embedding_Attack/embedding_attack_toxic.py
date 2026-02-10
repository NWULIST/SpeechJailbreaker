"""
Inspired by the llm-attacks project: https://github.com/llm-attacks/llm-attacks

Run:

    python embedding_attack_submission.py --help

for more information.
"""

import csv
import torch
import torch.nn as nn
import tqdm

from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

import torch
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
        return model.model.embed_tokens.weight

    else:
        raise ValueError(f"❌ Unknown model type: {type(model)}")



def generate(model, input_embeddings, num_tokens=50, input_features=None):
    """
    通用 embedding-level greedy 生成函数。
    支持：
      - GPT/LLAMA 等文本模型
      - Qwen2AudioForConditionalGeneration (可选音频输入)
    """
    model.eval()
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
    loss_slice_start = len(embeddings[0]) + len(embeddings_attack[0])
    logits_slice = logits[0, loss_slice_start - 1 : -1, :]
    targets_slice = targets[: logits_slice.size(0)]

    loss = nn.CrossEntropyLoss()(logits_slice, targets_slice)

    return loss, logits[:, loss_slice_start - 4 : -1, :]


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


def run(
    model_path: str,
    num_steps: int = 100,
    fixed_prompt: str = "I have a job to do.",
    control_prompt_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    target: str = "It seems like you're saying that you have something to do, but you're not specifying what that something is. Is that correct?",
    device: str = "cuda:0",
    num_tokens: int = 500,
    step_size: float = 0.01,
    print_interval: int = 5,
    generate_interval: int = 500,
    seed: int = 42,
    load_dataset=True,
    verbose=False,
    early_stopping=True,
):
    """
    Embedding space attack on Llama2.

    String will overall look like:

        [fixed_prompt] + [control_prompt] + [target]

                                                ^ target of optimization

                                ^ control tokens optimized to maximize target.
                                  genration begins at the end of these embeddings.

              ^ a fixed prompt that will not get modified during optimization. Can
                be used to provide a fixed context; matches the experimental setup
                of Zou et al., 2023.

    Args:
        model_path (str): Path to your Llama-2-7b-chat-hf directory
        num_steps (int): Number of gradient steps to take in the attack
        fixed_prompt (str): Part of the prompt that won't be altered/have gradients backpropogated
            to. You can specify an empty space i.e. fixed_prompt=' ' if you wish to only have a
            controllabe prompt.
        control_prompt (str): Part of the prompt that will be modified by gradient info. Generation
            starts at the end of this string.
        target (str): Optimization target; what the LLM will seek to generate immediately after
            the control string.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model, tokenizer = load_model_and_tokenizer(
        model_path, low_cpu_mem_usage=True, use_cache=False, device=device
    )
    embed_weights = get_embedding_matrix(model)

    if load_dataset:
        filename = "harmful_behaviors.csv"
        reader = csv.reader(open(filename, "r"))
        next(reader)
    else:
        print(f"Fixed prompt:\t '{fixed_prompt}'")
        print(f"Control prompt:\t '{control_prompt_init}'")
        print(f"Target string:\t '{target}'")
        reader = [[fixed_prompt, target]]

    total_steps = 0
    n = 0
    successful_attacks = 0

    for row in reader:
        fixed_prompt, target = row
        control_prompt = control_prompt_init
        print(fixed_prompt, target)

        # always appends a pad token at front; deal with it
        input_tokens = torch.tensor(tokenizer(fixed_prompt)["input_ids"], device=device)
        attack_tokens = torch.tensor(tokenizer(control_prompt)["input_ids"], device=device)[1:]
        target_tokens = torch.tensor(tokenizer(target)["input_ids"], device=device)[1:]

        # inputs
        one_hot_inputs, embeddings = create_one_hot_and_embeddings(input_tokens, embed_weights, model)
        # attack
        one_hot_attack, embeddings_attack = create_one_hot_and_embeddings(attack_tokens, embed_weights, model)
        # one_hot_attack, embeddings_attack = one_hot_attack[1:], embeddings_attack[1:]
        # targets
        one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_tokens, embed_weights, model)
        # one_hot_target, embeddings_target = one_hot_target[1:], embeddings_target[1:]

        adv_pert = torch.zeros_like(embeddings_attack, requires_grad=True, device=device)
        for i in range(num_steps):
            total_steps += 1
            loss, logits = calc_loss(
                model, embeddings, embeddings_attack + adv_pert, embeddings_target, one_hot_target
            )
            loss.backward()
            grad = adv_pert.grad.data
            adv_pert.data -= torch.sign(grad) * step_size

            model.zero_grad()
            adv_pert.grad.zero_()

            tokens_pred = logits.argmax(2)
            output_str = tokenizer.decode(tokens_pred[0][3:].cpu().numpy())
            sucess = output_str == target
            if sucess:
                successful_attacks += 1
                if early_stopping:
                    break

            if i % print_interval == 0 and i != 0:
                print(f"Iter: {i}")
                print(f"loss: {loss}")
                print(f"norms: {(embeddings_attack + adv_pert).norm(2, dim=2)}")
                print(f"output:{output_str}")

            if i % generate_interval == 0 and i != 0 and verbose:
                full_embedding = torch.hstack([embeddings, embeddings_attack + adv_pert])
                generated_tokens = generate(model, full_embedding, num_tokens)
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print("==============================================")
                print(generated_text)
                print("============================================== ")

        if verbose:
            full_embedding = torch.hstack([embeddings, embeddings_attack + adv_pert])
            generated_tokens = generate(model, full_embedding, num_tokens)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("================== FINAL =====================")
            print(generated_text)
            print("============================================== ")

        n += 1
        print(f"Successful attacks: {successful_attacks}/{n} \nAverage steps: {total_steps/n}")


if __name__ == "__main__":
    run()
