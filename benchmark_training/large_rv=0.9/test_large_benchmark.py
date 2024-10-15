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
parser = argparse.ArgumentParser(description="A script that processes input arguments.")
parser.add_argument('--resume-dir', type=str, default="checkpoint", help="Path to the resume directory")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
parser.add_argument('--data-idx', type=str, default="", help="Input the data index")
args = parser.parse_args()

with open("test_data/test_data.json","r") as file:
    data_list = json.load(file)

resume_dir = args.resume_dir
save_dir = args.resume_dir
for data in data_list:
#data = data_list[3] #0 - 59
    cfg = data["cfg"]
    net_xml = data["net_xml"]
    #wandb_id = extract_map_name(net_xml)+"xxx"
    rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    for rv_rate in rv_rate_list:
        wandb_name = extract_map_name(net_xml)+"("+extract_scenario(net_xml)+")_rv="+str(rv_rate)+"_bench=large"
        latest_checkpoint = get_latest_checkpoint(resume_dir)
        if args.wandb_id != "":
            command = [
                "python","SAC_eval.py",
                "--rv-rate",str(rv_rate),
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--wandb-id", args.wandb_id,
                "--wandb-name", wandb_name,
                "--resume-cp",str(latest_checkpoint)
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
        else:
            command = [
                "python","SAC_eval.py",
                "--rv-rate",str(rv_rate),
                "--num-cpu","2",
                "--cfg",str(cfg),
                "--map-xml",str(net_xml),
                "--wandb-id", "",
                "--wandb-name", wandb_name,
                "--resume-cp",str(latest_checkpoint)
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)