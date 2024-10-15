import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="A script that processes input arguments.")
parser.add_argument('--benchmark', type=str, default="0", help="Input the benchmark number")
parser.add_argument('--resume-dir', type=str, default="hybrid_checkpoint_new/benchmark_0", help="Path to the resume directory")
parser.add_argument('--wandb-id', type=str, default="", help="Input the wandb-id")
args = parser.parse_args()

# 定义 cfg_array 和 net_xml_array
cfg_array = [
    "files/outputs/ra_1_dict/ra_1.sumo.cfg",
    "files/outputs/ra_2_dict/ra_2.sumo.cfg",
    "files/outputs/ra_3_dict/ra_3.sumo.cfg",
    "files/outputs/test_3_dict/test_3.sumo.cfg",
    "files/outputs/int_0_dict/int_0.sumo.cfg",
    "real_data/osm.sumocfg"
]

net_xml_array = [
    "files/outputs/ra_1_dict/ra_1/ra_1.net.xml",
    "files/outputs/ra_2_dict/ra_2/ra_2.net.xml",
    "files/outputs/ra_3_dict/ra_3/ra_3.net.xml",
    "files/outputs/test_3_dict/test_3/test_3.net.xml",
    "files/outputs/int_0_dict/int_0/int_0.net.xml",
    "real_data/CSeditClean_1.net_threelegs.xml"
]
resume_dir = args.resume_dir
save_dir = "hybrid_checkpoint_new/benchmark_"+args.benchmark
if args.benchmark != "0":
    benchmark_sumocfg_path = "benchmark/benchmark_"+args.benchmark+"/benchmark_"+args.benchmark+"_sumocfg.txt"
    benchmark_net_path = "benchmark/benchmark_"+args.benchmark+"/benchmark_"+args.benchmark+"_net.txt"
    with open(benchmark_sumocfg_path, 'r') as f:
        new_list = [line.strip() for line in f]
        cfg_array = new_list + cfg_array

    with open(benchmark_net_path, 'r') as f:
        new_list = [line.strip() for line in f]
        net_xml_array = new_list + net_xml_array

#resume_dir = "hybrid_checkpoint"

def get_latest_checkpoint(resume_dir):
    checkpoint_dirs = [d for d in os.listdir(resume_dir) if d.startswith("checkpoint_")]
    checkpoint_dirs.sort()
    if checkpoint_dirs:
        return os.path.join(resume_dir, checkpoint_dirs[-1])
    else:
        return ""

# 外部循环
for outer_loop in range(100):
    # 内部循环
    for cfg, net_xml in zip(cfg_array, net_xml_array):
        latest_checkpoint = get_latest_checkpoint(resume_dir)
        if latest_checkpoint != "":
            command = [
                "python", "SAC_run.py",
                "--rv-rate", "1",
                "--stop-iters", "10",
                "--framework", "torch",
                "--num-cpu", "2",
                "--resume-cp", str(latest_checkpoint),
                "--save-dir", save_dir,
                "--cfg", str(cfg),
                "--map-xml", str(net_xml),
                "--wandb-id", args.wandb_id  # input the wandb id 
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
        else:
            command = [
                "python", "SAC_run.py",
                "--rv-rate", "1",
                "--stop-iters", "10",
                "--framework", "torch",
                "--num-cpu", "2",
                "--save-dir", save_dir,
                "--cfg", str(cfg),
                "--map-xml", str(net_xml),
                "--wandb-id", args.wandb_id  # input the wandb id 
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
        #     print(f"No checkpoints found in {resume_dir}, skipping.")