from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("NWULIST/AABench", "default")['train']  
origin_question = ds[0]
print(origin_question)