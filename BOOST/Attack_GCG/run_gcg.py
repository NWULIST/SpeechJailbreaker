import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for debugging
import pandas as pd
import argparse
from BOOST.Attack_GCG.gcg import GCG
import csv
from BOOST.utils.templates import get_eos

def gcg_attack(args):

    ds = load_dataset("MBZUAI/AudioJailbreak", "Origin")['origin']
    origin_question_audio = ds['speech_path'][args.index]
    origin_question_audio = origin_question_audio.replace("./audio", os.path.join(base_dir, "audio"))
    origin_question = ds['prompt'][args.index]
    question = origin_question
    target = pd.read_csv(args.targets_dataset)['target'].tolist()[args.index]


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