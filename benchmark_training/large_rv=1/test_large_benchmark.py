# import argparse
# import os
# import subprocess
# import json

# def get_latest_checkpoint(resume_dir):
#     checkpoint_dirs = [d for d in os.listdir(resume_dir) if d.startswith("checkpoint_")]
#     checkpoint_dirs.sort()
#     if checkpoint_dirs:
#         return os.path.join(resume_dir, checkpoint_dirs[-1])
#     else:
#         return ""
    
# def extract_map_name(file_path):
#     path_parts = file_path.split(os.sep)
#     map_name = path_parts[-2]
#     return map_name
    
# def extract_scenario(file_path):
#     path_parts = file_path.split(os.sep)
#     map_name = path_parts[-6]
#     return map_name
        
# def extract_mode(file_path):
#     path_parts = file_path.split(os.sep)
#     map_name = path_parts[-5]
#     return map_name
# parser = argparse.ArgumentParser(description="A script that processes input arguments.")
# parser.add_argument('--resume-dir', type=str, default="checkpoint_prev", help="Path to the resume directory")
# parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
# parser.add_argument('--data-idx', type=str, default="", help="Input the data index")
# args = parser.parse_args()

# with open("test_data/test_data.json","r") as file:
#     data_list = json.load(file)

# resume_dir = args.resume_dir
# save_dir = args.resume_dir
# #rv_rate_list = [1]
# rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# for count in range(2,6):
#     for i in range(1,len(data_list)): # 从29开始要弄0.4-0.9的内容
#         data = data_list[i] #0 - 59
#         #data = data_list[-10]
#         cfg = data["cfg"]
#         net_xml = data["net_xml"]
#         #wandb_id = extract_map_name(net_xml)+"xxx"
#         #rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#         for rv_rate in rv_rate_list:
#             wandb_name = extract_map_name(net_xml)+"("+extract_scenario(net_xml)+")_rv="+str(rv_rate)+"_bench=large_"+extract_mode(net_xml)+"_"+str(count)
#             latest_checkpoint = get_latest_checkpoint(resume_dir)
#             if args.wandb_id != "":
#                 command = [
#                     "python","SAC_eval.py",
#                     "--rv-rate",str(rv_rate),
#                     "--num-cpu","2",
#                     "--cfg",str(cfg),
#                     "--map-xml",str(net_xml),
#                     "--wandb-id", args.wandb_id,
#                     "--wandb-name", wandb_name,
#                     "--resume-cp",str(latest_checkpoint)
#                 ]
#                 print(f"Running command: {' '.join(command)}")
#                 subprocess.run(command)
#             else:
#                 command = [
#                     "python","SAC_eval.py",
#                     "--rv-rate",str(rv_rate),
#                     "--num-cpu","2",
#                     "--cfg",str(cfg),
#                     "--map-xml",str(net_xml),
#                     "--wandb-id", "",
#                     "--wandb-name", wandb_name,
#                     "--resume-cp",str(latest_checkpoint)
#                 ]
#                 print(f"Running command: {' '.join(command)}")
#                 subprocess.run(command)


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
parser.add_argument('--resume-dir', type=str, default="checkpoint_prev", help="Path to the resume directory")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
parser.add_argument('--data-idx', type=str, default="", help="Input the data index")
args = parser.parse_args()

# with open("test_data/test_data.json","r") as file:
#     data_list = json.load(file)
with open("test_data/test_data.json","r") as file:
    data_list = json.load(file)


resume_dir = args.resume_dir
save_dir = args.resume_dir
#rv_rate_list = [0.4]
# rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# checkpoint_list = ["checkpoint/checkpoint_000030_rv=0.4","checkpoint/checkpoint_000030_rv=0.5","checkpoint/checkpoint_000400_rv=0.6","checkpoint/checkpoint_000320_rv=0.7",
# "checkpoint/checkpoint_000400_rv=0.8","checkpoint/checkpoint_000030_rv=0.9","checkpoint/checkpoint_000400"]

rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9]
checkpoint_list = ["checkpoint/checkpoint_000130_rv=0.4","checkpoint/checkpoint_000320_rv=0.5","checkpoint/checkpoint_000400_rv=0.6","checkpoint/checkpoint_000320_rv=0.7",
"checkpoint/checkpoint_000400_rv=0.8","checkpoint/checkpoint_000320_rv=0.9"]

# rv_rate_list = [1.0, 1.0, 1.0, 1.0]
# checkpoint_list = ["checkpoint/checkpoint_000050","checkpoint/checkpoint_000100", "checkpoint/checkpoint_000200", "checkpoint/checkpoint_000300"]
# benchmark = ["50","100","200","300"]

# rv_rate_list = [1.0, 1.0, 1.0]
# checkpoint_list = ["checkpoint/checkpoint_000150","checkpoint/checkpoint_000250", "checkpoint/checkpoint_000350"]
# benchmark = ["150","250","350"]

# rv_rate_list = [1.0]
# checkpoint_list = ["checkpoint/checkpoint_000900"]
# benchmark = ["900"]



# rv_rate_list = [0.9]
# checkpoint_list = ["checkpoint/checkpoint_000320_rv=0.9"]
# rv_rate_list = [0.5]
# checkpoint_list = ["checkpoint/checkpoint_000320_rv=0.5"]
# rv_rate_list = [1.0]
# checkpoint_list = ["checkpoint/checkpoint_000400"]

# for count in range(4,5):
#for i in range(-10,-5): # 从29开始要弄0.4-0.9的内容
# 第3个终端是0-9； 第4个终端是10-19； 第一个终端是20-53； 第2个终端是在做BENCHMARK SIZE的test
for i in range(20,len(data_list)):
    data = data_list[i] #0 - 59
    #data = data_list[-10]
# map-11 middle -> -8
# map-11 hard -> -16
# map-5 middle -> -4
# map-5 hard -> -12

#data = data_list[-4]
    cfg = data["cfg"]
    net_xml = data["net_xml"]
    #wandb_id = extract_map_name(net_xml)+"xxx"
    #rv_rate_list = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for j in range(len(rv_rate_list)):
        #wandb_name = extract_map_name(net_xml)+"("+extract_scenario(net_xml)+")_rv="+str(rv_rate)+"_bench=large_"+extract_mode(net_xml)+"_"+str(count)
        # map-16(intersection)_rv=0.5_easy_new
        rv_rate = rv_rate_list[j]
        latest_checkpoint = checkpoint_list[j]
        # wandb_name = extract_map_name(net_xml)+"("+extract_scenario(net_xml)+")_rv="+str(rv_rate)+"_"+extract_mode(net_xml)+"_new_"+benchmark[j]
        wandb_name = extract_map_name(net_xml)+"("+extract_scenario(net_xml)+")_rv="+str(rv_rate)+"_"+extract_mode(net_xml)+"_new_2"
        #latest_checkpoint = get_latest_checkpoint(resume_dir)
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