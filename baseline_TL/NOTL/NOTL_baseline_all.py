import argparse
import os
import subprocess
import json

def extract_map_name(file_path):
    path_parts = file_path.split(os.sep)
    map_name = path_parts[-2]
    return map_name
def extract_mode(file_path):
    path_parts = file_path.split(os.sep)
    mode_name = path_parts[-5]
    return mode_name

parser = argparse.ArgumentParser(description="A script that processes input arguments.")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
args = parser.parse_args()

# with open("test_data/test_data.json","r") as file:
#     data_list = json.load(file)

# with open("test_data/test_data_400.json","r") as file:
with open("test_data/test_data.json","r") as file:
    data_list = json.load(file)

for i in range(len(data_list)):
    data = data_list[i]
    cfg = data["cfg"]
    net_xml = data["net_xml"]
    wandb_name = extract_map_name(net_xml)+"_"+extract_mode(net_xml) + "_NOTL_speedmode=7"
    #for i in range(5):
    if args.wandb_id != "":
        command = [
            "python","NOTL_main.py",
            "--cfg",str(cfg),
            "--map-xml",str(net_xml),
            "--wandb-name", wandb_name,
            "--wandb-id", args.wandb_id
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)
    else:
        command = [
            "python","NOTL_main.py",
            "--cfg",str(cfg),
            "--map-xml",str(net_xml),
            "--wandb-name", wandb_name
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)