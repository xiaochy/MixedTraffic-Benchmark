import argparse
import os
import subprocess
import json

def get_latest_checkpoint(resume_dir):
    checkpoint_dirs = [d for d in os.listdir(resume_dir) if d.startswith("checkpoint_")]
    checkpoint_dirs.sort()
    if checkpoint_dirs:
        return os.path.join(resume_dir, checkpoint_dirs[-1])
    else:
        return ""

parser = argparse.ArgumentParser(description="A script that processes input arguments.")
parser.add_argument('--resume-dir', type=str, default="checkpoint", help="Path to the resume directory")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
args = parser.parse_args()

with open("train_data/train_data.json","r") as file:
    data_list = json.load(file)

resume_dir = args.resume_dir
save_dir = args.resume_dir
for outer_loop in range(100):
    for data in data_list:
        cfg = data["cfg"]
        net_xml = data["net_xml"]
        junction_list = data["junction_list"]
        latest_checkpoint = get_latest_checkpoint(resume_dir)
        if latest_checkpoint != "":
            command = [
                "python","DQN_run.py",
                "--rv-rate","1",
                "--stop-iters","10",
                "--framework","torch",
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--junction-list",str(junction_list),
                "--wandb-id", args.wandb_id,
                "--save-dir", save_dir,
                "--resume-cp",str(latest_checkpoint)
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
        else:
            command = [
                "python","DQN_run.py",
                "--rv-rate","1",
                "--stop-iters","10",
                "--framework","torch",
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--junction-list",str(junction_list),
                "--wandb-id", args.wandb_id,
                "--save-dir", save_dir
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)