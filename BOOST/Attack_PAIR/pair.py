import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import csv
import pandas as pd
from BOOST.Attack_GPTFuzzer.gptfuzzer.llm import OpenAILLM, LocalSpeechLLM, OpenAIAudioLLM
#from BOOST.Attack_GPTFuzzer.gptfuzzer.llm import LocalLLM, ClaudeLLM, GeminiLLM
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)
from strongreject.strongreject_evaluator import autograde_response
from typing import NamedTuple
from datasets import load_dataset
import json
#from loggers import WandBLogger
from Defenses.SmoothLLM.smoothllmWrapper import smoothllmWrapper

import argparse
#from judges import load_judge
from conversers import load_attack_and_target_models
from common import process_target_response, initialize_conversations
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import re



ds = load_dataset("NWULIST/AABench", "default")['train']  


def memory_usage_psutil():
    # Returns the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)  # bytes to MB
    return mem

class EvalResult(NamedTuple):
    success: bool
    score: float

def evaluate_generation_strongreject(question, generation):
    result = autograde_response(question, generation)
    success = result.score_zero_to_one is not None and result.score_zero_to_one > 0
    score = result.score_zero_to_one if result.score_zero_to_one is not None else 0.0
    return EvalResult(success, score)

def PAIR_attack(args, base_dir = "/projects/e33046/AABench"):
    memory_before = memory_usage_psutil()


    if getattr(args, 'store_folder', None):
        os.makedirs(args.store_folder, exist_ok=True)


    system_message = None
    if args.defence == 'adashield':
        base_path = os.path.expanduser('~') 
        defense_prompt_path = os.path.join(base_path, 'SpeechJailbreaker', 'Defense_prompt', 'adashield.json')

        if os.path.isfile(defense_prompt_path):
            with open(defense_prompt_path, "r") as f:
                system_message = json.load(f)["prompt"]
                print(system_message)
        else:
            raise FileNotFoundError(f"Defense file not found: {defense_prompt_path}")
    elif args.defence == 'smoothllm':
        print('smoothllm defense, passing for now')
    elif args.defence != '' and args.defence != 'guard' and args.defence != 'None':
        print('passing')
        pass
    else:
        print("No Defense")
    


    if 'gpt' in args.target_model and 'audio' not in args.target_model:
        print(args.target_model)
        target_model = OpenAILLM(args.target_model, args.openai_key, system_message=system_message)
    elif 'claude' in args.target_model:
        target_model = ClaudeLLM(args.target_model, args.claude_key)
    elif 'gemini' in args.target_model:
        target_model = GeminiLLM(args.target_model, args.gemini_key)
    elif 'gpt' in args.target_model.lower() and 'audio' in args.target_model.lower():
        print(args.target_model)
        target_model = OpenAIAudioLLM(args.target_model, args.openai_key, system_message=system_message)
    elif 'audio' in args.target_model.lower():
        print(args.target_model)
        target_model = LocalSpeechLLM(args.target_model, system_message=system_message)
    else:
        target_model = LocalLLM(args.target_model, system_message=system_message)

    if args.defence == "smoothllm":
        base_model = target_model
        selected_pert = "RandomSwapPerturbation"
        selected_num_copies = args.num_copies
        selected_pert_pct = 0.15
        target_model = smoothllmWrapper(base_model, pert_type=selected_pert, pert_pct=selected_pert_pct, num_copies=selected_num_copies)
        print(f"[SMOOTHLLM] Applying {selected_pert} with {selected_num_copies} num_copies and {selected_pert_pct}")


    ########EVALUATION#############################################################################

    evaluation = getattr(args, 'evaluation', 'default')
    print("The evaluation is: ", evaluation)

    #save strong reject scores
    strongreject_save_path = f'./Results/{args.target_model}/PAIR/PAIR_{args.run_identifier}.csv'
    sr_csv_file = open(strongreject_save_path, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(sr_csv_file)

   
    # Initialize attack and target models
    attackLM, _ = load_attack_and_target_models(args)

    #get indices part of this batch
    indices_to_process = args.indices_list

    # Begin PAIR

    for idx in indices_to_process:
        print(f"\n{'='*60}")
        print(f"Processing index {idx}")
        print(f"{'='*60}")

        relative_path = ds['prompt_audio_path'][idx]
        origin_question_audio = os.path.join(base_dir, relative_path)
        origin_question = ds['prompt_text'][idx]

            # Initialize conversations
        print(attackLM.template)
        print(origin_question)
        convs_list, processed_response_list, system_prompts = initialize_conversations(args.n_streams, origin_question, args.target_str, attackLM.template)
        target_response_list, judge_scores = None, None


        if args.defence == 'self-reminder':
            system_message = system_message.replace("{QUESTION}", origin_question)
            print(system_message)


        for iteration in range(0, args.n_iterations):
            print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
            print("iteration: ", iteration)

            #if iteration > 0:
            #    processed_response_list = [process_target_response(target_response, origin_question, args.target_str) for target_response in zip(target_response_list, judge_scores)]

            # Get adversarial prompts and improvement
            extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
            if not extracted_attack_list:
                print("Failed to generate an attack. Aborting this prompt")
                csv_writer.writerow([idx, origin_question, iteration, "Failed to generate an attack.", "Failed to generate an attack.", 0, "Aborted"])
                sr_csv_file.flush()
                break

            # Extract prompts and improvements
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            improv_list = [attack["improvement"] for attack in extracted_attack_list]

            #memory check
            memory_after = memory_usage_psutil()
            print(f"Memory before: {memory_before} MB")
            print(f"Memory after: {memory_after} MB")

            #get base model reference if wrapped
            model_ref = getattr(target_model, "base_model", target_model)

            # Get target responses
            # LocalSpeechLLM requires both prompts (audio paths) and texts
            try:
                if isinstance(model_ref, LocalSpeechLLM):
                    # For audio models, use origin_question_audio for prompts and adv_prompt_list for texts
                    audio_prompts = [origin_question_audio] * len(adv_prompt_list)
                    target_response_list = target_model.generate_batch(audios=audio_prompts, prompts=adv_prompt_list, max_tokens=512)
                elif isinstance(target_model, OpenAIAudioLLM):
                    # For audio models, use origin_question_audio for prompts and adv_prompt_list for texts
                    audio_prompts = [origin_question_audio] * len(adv_prompt_list)
                    target_response_list = target_model.generate_batch(prompts=adv_prompt_list, audios=audio_prompts,  max_tokens=512)
                elif isinstance(model_ref, OpenAIAudioLLM):
                    audio_prompts = [origin_question_audio] * len(adv_prompt_list)
                    target_response_list = target_model.generate_batch(audios=audio_prompts, prompts=adv_prompt_list, max_tokens=512)
                else:
                    # For other models, use generate_batch with text prompts
                    target_response_list = target_model.generate_batch(adv_prompt_list, max_tokens=512)

                print("Finished getting target responses.")
                
            except (ValueError,FileNotFoundError) as e:
                print(f"Skipping index {idx}: audio file error {e}")
                csv_writer.writerow([idx, origin_question, 0, "N/A", "N/A", f"Skipped: {e}", 0, False])
                sr_csv_file.flush()
                continue
            
            # Get judge scores using StrongReject
            judge_scores = []
            any_success = False
            
            for stream_idx, response in enumerate(target_response_list):
                # Evaluate this specific stream
                result = evaluate_generation_strongreject(origin_question, response)
                print(f"Stream {stream_idx} | StrongReject Score: {result.score:.4f}, Success: {result.success}", flush=True)
                
                # Log it
                csv_writer.writerow([idx, origin_question, iteration, stream_idx, adv_prompt_list[stream_idx], response, result.score, result.success])
                sr_csv_file.flush()
                
                scaled_score = result.score * 10
                judge_scores.append(scaled_score)
                
                if result.success:
                    any_success = True
            
            # check if we break or continue
            if any_success:
                print("Jailbreak successful on at least one stream!")
                break
            
            # Since we didn't break, prepare the feedback for the next loop.
            processed_response_list = [
                process_target_response(
                        target_response=target_response, 
                        score=score,
                        goal=origin_question,
                        target_str=args.target_str
                    ) 
                    for target_response, score in zip(target_response_list, judge_scores)
            ]

    sr_csv_file.close()

