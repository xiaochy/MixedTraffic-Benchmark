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
        lane_entry = data["lane_entry"]
        junction_id = data["junction_id"]
        latest_checkpoint = get_latest_checkpoint(resume_dir)
        if latest_checkpoint != "":
            command = [
                "python","PPO_ra_run.py",
                "--rv-rate","0",
                "--stop-iters","10",
                "--framework","torch",
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--lane-entry",str(lane_entry),
                "--junction-id",str(junction_id),
                "--wandb-id", args.wandb_id,
                "--save-dir", save_dir,
                "--resume-cp",str(latest_checkpoint)
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
        else:
            command = [
                "python","PPO_ra_run.py",
                "--rv-rate","0",
                "--stop-iters","10",
                "--framework","torch",
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--lane-entry",str(lane_entry),
                "--junction-id",str(junction_id),
                "--wandb-id", args.wandb_id,
                "--save-dir", save_dir
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)