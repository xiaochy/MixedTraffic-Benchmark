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
    
# with open("train_data_small/train_data.json","r") as file:
#     data_list = json.load(file)

resume_dir = args.resume_dir
save_dir = args.resume_dir
for outer_loop in range(100):
    for i in range(len(data_list)):
        data = data_list[i]
        cfg = data["cfg"]
        net_xml = data["net_xml"]
        latest_checkpoint = get_latest_checkpoint(resume_dir)
        if latest_checkpoint != "":
            for i in range(10):
                save = False
                if i == 9:
                    save = True
                command = [
                    "python","SAC_run.py",
                    "--rv-rate","1",
                    "--stop-iters","10",
                    "--framework","torch",
                    "--num-cpu","2",
                    "--cfg",str(cfg),
                    "--map-xml",str(net_xml),
                    "--wandb-id", args.wandb_id,
                    "--save-dir", save_dir,
                    "--resume-cp",str(latest_checkpoint),
                    "--save", str(save)
                ]
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)
        else:
            for i in range(10):
                save = False
                if i == 9:
                    save = True
                command = [
                    "python","SAC_run.py",
                    "--rv-rate","1",
                    "--stop-iters","10",
                    "--framework","torch",
                    "--num-cpu","2",
                    "--cfg",str(cfg),
                    "--map-xml",str(net_xml),
                    "--wandb-id", args.wandb_id,
                    "--save-dir", save_dir,
                    "--save", str(save)
                ]
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)