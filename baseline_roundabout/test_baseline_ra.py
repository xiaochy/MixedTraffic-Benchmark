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
    
def extract_map_name(file_path):
    path_parts = file_path.split(os.sep)
    map_name = path_parts[-2]
    return map_name

def extract_scenario(file_path):
    path_parts = file_path.split(os.sep)
    map_name = path_parts[-6]
    return map_name

def extract_mode(file_path):
    path_parts = file_path.split(os.sep)
    map_name = path_parts[-5]
    return map_name
parser = argparse.ArgumentParser(description="A script that processes input arguments.")
parser.add_argument('--resume-dir', type=str, default="checkpoint", help="Path to the resume directory")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
args = parser.parse_args()

with open("test_data/test_data.json","r") as file:
    data_list = json.load(file)


resume_dir = args.resume_dir
save_dir = args.resume_dir
for data in data_list:
    cfg = data["cfg"]
    net_xml = data["net_xml"]
    junction_id = data["junction_id"]
    lane_entry = data["lane_entry"]
    wandb_name = extract_map_name(net_xml)+"_"+extract_mode(net_xml)+"("+extract_scenario(net_xml)+")_baseline"
    latest_checkpoint = get_latest_checkpoint(resume_dir)
    if args.wandb_id != "":
        command = [
            "python","PPO_ra_eval.py",
            "--rv-rate","1",
            "--num-cpu","2",
            "--cfg",str(cfg),
            "--map-xml",str(net_xml),
            "--junction-id",str(junction_id),
            "--lane-entry",str(lane_entry),
            "--wandb-id", args.wandb_id,
            "--wandb-name", wandb_name,
            "--resume-cp",str(latest_checkpoint)
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)
    else:
        command = [
            "python","PPO_ra_eval.py",
            "--rv-rate","1",
            "--num-cpu","2",
            "--cfg",str(cfg),
            "--map-xml",str(net_xml),
            "--junction-id",str(junction_id),
            "--lane-entry",str(lane_entry),
            "--wandb-id", "",
            "--wandb-name", wandb_name,
            "--resume-cp",str(latest_checkpoint)
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)