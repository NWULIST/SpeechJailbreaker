import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import pandas as pd
import argparse
from BOOST.Attack_GCG.gcg import GCG
import csv
try:
    from strongreject.strongreject_evaluator import autograde_response
    STRONGREJECT_AVAILABLE = True
except ImportError:
    STRONGREJECT_AVAILABLE = False
    print("Warning: strongreject not available. Install it to use strongreject evaluation.")



class EvalResult(NamedTuple):
    success: bool
    score: float

def evaluate_generation_strongreject(question, generation, target):
    result = autograde_response(question, generation)
    success = result.score_zero_to_one is not None and result.score_zero_to_one > 0
    score = result.score_zero_to_one if result.score_zero_to_one is not None else 0.0
    return EvalResult(success, score)



def pgd_attack(args):

    ds = load_dataset("MBZUAI/AudioJailbreak", "Origin")['origin']
    origin_question_audio = ds['speech_path'][args.index]
    origin_question_audio = origin_question_audio.replace("./audio", os.path.join(base_dir, "audio"))
    origin_question = ds['prompt'][args.index]
    origin_question_audio = ds['speech_path'][args.index]
    target = pd.read_csv(args.targets_dataset)['target'].tolist()[args.index]

    system_message = None
    if args.defence != '' and args.defence != 'guard' and args.defence != "None":
        defence_path = f"/projects/e33046/AttackBench/Defense_prompt/{args.defence}.json"
        print(defence_path)

        if os.path.isfile(defence_path):
            with open(defence_path, "r") as f:
                system_message = json.load(f)["prompt"]
        else:
            raise FileNotFoundError(f"Defense file not found: {defence_path}")


    if args.defence == 'self-reminder':
        system_message = system_message.replace("{QUESTION}", origin_question)
        print(system_message)

    if 'gpt' in args.target_model:
        target_model = OpenAILLM(args.target_model, args.openai_key)
    elif 'claude' in args.target_model:
        target_model = ClaudeLLM(args.target_model, args.claude_key)
    elif 'gemini' in args.target_model:
        target_model = GeminiLLM(args.target_model, args.gemini_key)
    elif 'audio' in args.target_model.lower():
        target_model = LocalSpeechLLM(args.target_model, system_message=system_message)
    else:
       target_model = LocalLLM(args.target_model, system_message=system_message)
    


    args.question = question
    print("args.index: ", args.index)
    print("The question is: ", question)
    print("The target is: ", target)
    
    gcg = GCG(args)
    optim_prompts, steps, scores = gcg.run(target)
    
    # save the optim prompts into a csv file
    save_path = f'./Results/{args.model_path}/GCG-{args.run_index}/{args.index}.csv'
    if args.add_eos:
        save_path = f'./Results/{args.model_path}/GCG_eos-{args.run_index}/{args.index}.csv'
    # Add evaluation method as a folder
    evaluation = getattr(args, 'evaluation', 'default')
    save_path = save_path.replace(f'/{args.index}.csv', f'/{evaluation}/{args.index}.csv')
        
    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        #write the column name
        if evaluation == 'strongreject':
            writer.writerow(['optim_prompts', 'steps', 'scores'])
            for prompt, step, score in zip(optim_prompts, steps, scores):
                writer.writerow([prompt, step, score])
        else:
            writer.writerow(['optim_prompts', 'steps'])
            for prompt, step in zip(optim_prompts, steps):
                writer.writerow([prompt, step])
    
    print("The optim prompts are saved.")