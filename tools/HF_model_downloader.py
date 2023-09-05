import os
import sys
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='facebook/wav2vec2-base-960h', help="model name")
args = parser.parse_args()

model_name = args.model_name
model_save_path = f"models/HF_models/{model_name.split('/')[-1]}"
huggingface_model_hub_url = "https://huggingface.co/" + model_name 
print(huggingface_model_hub_url)
if not os.path.exists(model_save_path):
    os.system("git lfs install")
    try:
        os.system(f"GIT_LFS_SKIP_SMUDGE=1 git clone {huggingface_model_hub_url}" )
        time.sleep(1)
    except:
        print("git clone failed")
        sys.exit(1)


os.chdir(model_save_path)
# os.system(f"cd {model_save_path}")
os.system(f"git lfs pull --include='*.bin' ")
