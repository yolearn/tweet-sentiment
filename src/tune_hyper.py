import subprocess
subprocess.Popen("ls")
subprocess.Popen("CUDA_VISIBLE_DEVICES=1 python3 train.py --MODEL_TYPE roberta --MODEL_VERSION roberta-base --MODEL_NAME robert_baseline")