from typing import Set
import random, math
from copy import deepcopy
import numpy as np
import wandb
import itertools
import gymnasium as gym
from ray.rllib.utils.typing import AgentID
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import traci  as T
from gymnasium.spaces import Discrete
from gymnasium.spaces.box import Box
import core
from core.monitor import DataMonitor
from core.sumo_interface import SUMO
from core.costomized_data_structures import Vehicle, Container
from core.NetMap import NetMap
# need to delete, we cannot use the global const as input
from core.utils import dict_tolist
import pdb

WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
EPSILON = 0.00001
# max number of lanes to be considered for a junction
# MAX_NUM_LANE = 20
FRONT_VEH_NUM = 10
BACK_VEH_NUM = 5
FRONT_DISTANCE = 50
BACK_DISTANCE = 20
WIDTH = 10
MAX_FRONT_SPEED = 25
MAX_BACK_SPEED = 35
# need to change !!!  # self.rv_num * 5 + self.junction_num * self.junction_veh_num * 3
#BASELINE_OBS_LENGTH = 3*2+3*2+3*6+3*6*2+3*1 



class Env(gym.Env):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.print_debug = False
        self.cfg = config['cfg']
        self.map_xml = config['map_xml']
        self._max_episode_steps = 1000
        if 'max_episode_steps' in config.keys():
            self._max_episode_steps = self.config['max_episode_steps']
        self.traffic_light_program = self.config['traffic_light_program']
        self.spawn_rl_prob = config['spawn_rl_prob']
        self.default_rl_prob = config['probablity_RL']
        #self.rl_prob_list = config['rl_prob_range'] if 'rl_prob_range' in config.keys() else None
        self.sumo_interface = SUMO(self.cfg, render=self.config['render'])
        self.map = NetMap(self.map_xml)
        self.junction_list = self.map.junction_list
        self.junction_data = self.map.junction_data
        self.max_acc = 10
        self.min_acc = -10
        self.max_speed = 100   
        self.vehicle_length = 5  
        self.alpha = 1
        self.beta = 2
        self.gamma = 5
        self.a = -1.5
        self.b = -1
        #self.n_obs = BASELINE_OBS_LENGTH
        self.max_action_dim = 10
        self.junction_veh_num = 6
        self.veh_count = 0
        self.restrict_speed = 15
        if "wandb_id" in config.keys():
            self.wandb_id = config["wandb_id"]
        if "wandb_name" in config.keys():
            self.wandb_name = config["wandb_name"]
        self.lane_entry_list = config["lane_entry"]
        self.lane_entry = {}
        for lane in config["lane_entry"]:
            self.lane_entry[lane] = {}
            self.lane_entry[lane]["entry_queue_length"] = 0
            self.lane_entry[lane]["front_veh_id"] = ""
            self.lane_entry[lane]["max_front_distance"] = 0
        self.junction_id = config["junction_id"]
        self.junction_position = []
        self.rv_num = len(self.lane_entry)
        #self.n_obs = 5*self.rv_num + 3*len(self.junction_id)*self.junction_veh_num
        self.n_obs = 5*self.max_action_dim+3*self.max_action_dim*self.junction_veh_num
        self.action_space = Box(low=self.min_acc, high=self.max_acc, shape=(self.max_action_dim,),dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(self.n_obs,),dtype=np.float32)
        self.init_env()


    # helper function
    # width corresponds to the edge parallel to the midline passing through the origin,
    # while height corresponds to the edge perpendicular to the midline.
    def is_point_in_rotated_rectangle(self, px, py, cx, cy, width, height, angle):
        """
        Determine if the point (px, py) is inside the rotated rectangle

        Parameters:
        px, py: Coordinates of the point
        cx, cy: Coordinates of the rectangle's center point
        width: Width of the rectangle
        height: Height of the rectangle
        angle: Rotation angle of the rectangle (in radians)

        Returns:
        True if the point is inside the rectangle, otherwise False
        """
        # Calculate the offset of the point relative to the rectangle center
        dx = px - cx
        dy = py - cy
        # Rotate the point to the rectangle's coordinate system
        rotated_x = dx * math.cos(-angle) - dy * math.sin(-angle)
        rotated_y = dx * math.sin(-angle) + dy * math.cos(-angle)
        # Check if the rotated point is inside the axis-aligned rectangle
        half_width = width / 2
        half_height = height / 2
        return (-half_width <= rotated_x <= half_width) and (-half_height <= rotated_y <= half_height)
    
    def rotate_vector(self,vector, radian):
        # Calculate the rotation angle
        angle = np.pi / 2 - radian
        # Define the rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        # Apply the rotation matrix to the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector


    def wait_time_reward(self, avg_wait_time):
        if 20 <= avg_wait_time <= 30:
            return 1  
        else:
            return -abs(avg_wait_time - 25) / 25


    def collision_reward(self,collision_number):
        if collision_number < 1:
            return 1 
        else:
            return -collision_number 

    def throughput_reward(self,throughput):
        return throughput
    
    def init_env(self):
        self.vehicles = Container()
        self.rl_vehicles = Container()
        self.veh_name_mapping_table = dict()
        self._step = 0
        self.previous_obs = {}  
        self.previous_dones = {}
        self.junction_throughput = []
        self.collision_num = []
        self.reward_record = []
        self.avg_waiting_time = []
        for junction_id in self.junction_id:
            self.junction_position.append(np.array(self.sumo_interface.tc.junction.getPosition(junction_id)))
        self._print_debug('init_env')


    def _update_obs(self,action):
        reward = 0
        avg_waiting = 0
        num_veh_speed_0 = 0
        num_veh_speed_0_3 = 0
        # inside the sqrt of the reward
        tmp_reward = 0
        total_num_veh = len(self.vehicles)

        obs = []
        dones = False
        position_dict = {}
        velocity_dict = {}
        speed_dict = {}
        angle_radian_dict = {}
        direction_dict = {}
        veh_lane_dict = {}
        front_distance_dict = {}

        rv_entry_velocity = []
        rv_entry_position = []
        # entry_queue_length = np.array([0 for i in range(self.rv_num)])
        # rv_entry_velocity = np.array([[0,0] for i in range(self.rv_num)])
        # rv_entry_position = np.array([[0,0] for i in range(self.rv_num)])
        entry_queue_length = np.array([0 for i in range(self.max_action_dim)])
        rv_entry_velocity = np.array([[0,0] for i in range(self.max_action_dim)])
        rv_entry_position = np.array([[0,0] for i in range(self.max_action_dim)])
        for lane, lane_dict in self.lane_entry.items():
            lane_dict["entry_queue_length"] = 0
            lane_dict["front_veh_id"] = ""
            lane_dict["max_front_distance"] = 0
        # nearest_veh_distance = np.array([[-1 for _ in range(self.junction_veh_num)] for _ in range(self.rv_num)])
        # nearest_veh_velocity = np.array([[[0,0] for _ in range(self.junction_veh_num)] for _ in range(self.rv_num)]) 
        nearest_veh_distance = np.array([[-1 for _ in range(self.junction_veh_num)] for _ in range(self.max_action_dim)])
        nearest_veh_velocity = np.array([[[0,0] for _ in range(self.junction_veh_num)] for _ in range(self.max_action_dim)]) 
        # number of vehicles with speed = 0 and speed < 0.3
        for veh in self.vehicles:
            # check wait_time
            wait_time, accum_wait_time = self.sumo_interface.get_veh_waiting_time(veh)
            avg_waiting += wait_time
            position_dict[veh.id] = self.sumo_interface.get_position(veh)
            velocity_dict[veh.id], angle_radian_dict[veh.id], direction_dict[veh.id] = self.sumo_interface.get_velocity_angle_direction(veh)
            speed_dict[veh.id] = self.sumo_interface.get_speed(veh)
            veh_lane_dict[veh.id] = self.sumo_interface.get_lane(veh)
            front_distance_dict[veh.id] = self.sumo_interface.get_lane_position(veh)
            if speed_dict[veh.id] == 0:
                num_veh_speed_0 += 1
            if speed_dict[veh.id] < 0.3:
                num_veh_speed_0_3 += 1
            tmp_reward += (speed_dict[veh.id]-self.restrict_speed)**2
            if veh_lane_dict[veh.id] in self.lane_entry.keys():
                self.lane_entry[veh_lane_dict[veh.id]]["entry_queue_length"] += 1
                if front_distance_dict[veh.id] > self.lane_entry[veh_lane_dict[veh.id]]["max_front_distance"]:
                    self.lane_entry[veh_lane_dict[veh.id]]["max_front_distance"] = front_distance_dict[veh.id]
                    self.lane_entry[veh_lane_dict[veh.id]]["front_veh_id"] = veh.id
                
                    
        if len(self.vehicles) != 0:
            avg_waiting /= len(self.vehicles)
        wandb.log({"average_waiting_time": avg_waiting})
        # check self.avg_waiting_time
        self.avg_waiting_time.append(avg_waiting)
        reward_my = self.alpha * self.throughput_reward(self.junction_throughput[-1]) + self.beta * self.collision_reward(self.collision_num[-1]) + self.gamma*self.wait_time_reward(avg_waiting)
        wandb.log({"reward_my": reward_my})
        # the velocity and position of the vehicles on the entry points
        #for index, lane_dict in enumerate(self.lane_entry):
        for lane, lane_dict in self.lane_entry.items():
            index = self.lane_entry_list.index(lane)
            if lane_dict["front_veh_id"] != "":
                rv_entry_velocity[index] = velocity_dict[lane_dict["front_veh_id"]]
                rv_entry_position[index] = position_dict[lane_dict["front_veh_id"]]
                entry_queue_length[index] = lane_dict["entry_queue_length"]
        # nearest vehicle to the three junctions (distance and velocity)
        #distance_to_junction = [[] for i in range(self.rv_num)]
        distance_to_junction = [[] for i in range(self.max_action_dim)]
        for i in range(self.rv_num):
            distance_to_junction[i] = [(veh.id, self.sumo_interface.calculate_distance(position_dict[veh.id],self.junction_position[i])) for veh in self.vehicles]
            distance_to_junction[i].sort(key=lambda x: x[1])
            length = len(distance_to_junction[i])
            if length >= self.junction_veh_num:
                length = self.junction_veh_num
            for j in range(length):
                nearest_veh_distance[i][j] = distance_to_junction[i][j][1]
                nearest_veh_velocity[i][j] = velocity_dict[distance_to_junction[i][j][0]]
        obs_tmp = []
        # self.rv_num * 5 + self.junction_num * self.junction_veh_num * 3  3*5+3*6*3 = 15+54 = 69 
        obs_tmp.append(list(self.normalize_array(rv_entry_velocity)))
        obs_tmp.append(list(self.normalize_array(rv_entry_position)))
        obs_tmp.append(list(self.normalize_array(nearest_veh_distance)))
        obs_tmp.append(list(self.normalize_array(nearest_veh_velocity)))
        obs_tmp.append(list(self.normalize_array(entry_queue_length)))
        flattened_obs = np.hstack([np.ravel(arr) for arr in obs_tmp])        
        obs = flattened_obs
        rewards = self.reward_compute_global(num_veh_speed_0, num_veh_speed_0_3, total_num_veh, tmp_reward)
        #wandb.log({"reward": rewards})
        dones = False   
        return obs, rewards, dones
    
    def reward_compute_global(self,num_veh_speed_0, num_veh_speed_0_3, total_num_veh, tmp_reward):
            reward = self.a*num_veh_speed_0 + self.b*num_veh_speed_0_3
            if self.restrict_speed*math.sqrt(total_num_veh) != 0:
                tmp = max(self.restrict_speed*math.sqrt(total_num_veh) - math.sqrt(tmp_reward),0)/(self.restrict_speed*math.sqrt(total_num_veh))
            else:
                tmp = 0
            reward += tmp 
            self.reward_record.append(reward)
            return reward
    
    def normalize_array(self,arr):
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val == min_val:
                return np.zeros_like(arr)
            normalized_arr = 2 * (arr - min_val) / (max_val - min_val) - 1
            return normalized_arr
        
    # def step_once(self, action={}):
    def step_once(self, action=[]):
        #print("action = ", action)
        self.new_departed = set()
        self.sumo_interface.set_max_speed_all(self.max_speed)
        self._traffic_light_program_update()
        # apply the policy action to the three vehicles on the three entry points
        rv_list = []
        #print("lane_entry = ", self.lane_entry)
        #for index, lane_dict in enumerate(self.lane_entry):
        for lane, lane_dict in self.lane_entry.items():
            index = self.lane_entry_list.index(lane)
            #print("lane_dict = ",lane_dict)
            if lane_dict["front_veh_id"] != "":
                if lane_dict["front_veh_id"] in self.vehicles.keys():
                    self.sumo_interface.accl_control(self.vehicles[lane_dict["front_veh_id"]], action[index])
                    print("apply_action")
                    rv_list.append(lane_dict["front_veh_id"])

        for veh_id, _ in self.vehicles.items():
            #if veh_id not in self.rl_vehicles: # need to change
            if veh_id not in rv_list:
                if random.random() < 0.5:
                    self.sumo_interface.accl_control(self.vehicles[veh_id], self.max_acc)
                else:
                    self.sumo_interface.accl_control(self.vehicles[veh_id], self.min_acc)

        # sumo simulation step once
        self.sumo_interface.step()
        # gathering states from sumo 
        _, sim_res = self.sumo_interface.get_sim_info()
        # setup for new departed vehicles 
        for veh_id in sim_res.departed_vehicles_ids:
            self.sumo_interface.subscribes.veh.subscribe(veh_id)
            length = self.sumo_interface.get_vehicle_length(veh_id)
            self.sumo_interface.tc.vehicle.setLength(veh_id, self.vehicle_length)
            length = self.vehicle_length
            route = self.sumo_interface.get_vehicle_route(veh_id)
            road_id  = self.sumo_interface.get_vehicle_edge(veh_id)
            if (road_id in self.spawn_rl_prob.keys() and random.random()<self.spawn_rl_prob[road_id]) or \
                (random.random()<self.default_rl_prob):
                self.rl_vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length)
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length, wait_time=0)
                self.sumo_interface.tc.vehicle.setSpeedMode(veh_id,39)
            else:
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="IDM", route=route, length=length, wait_time=0)
                self.sumo_interface.tc.vehicle.setSpeedMode(veh_id,39)
            self.sumo_interface.set_color(veh, WHITE if veh.type=="IDM" else RED)
            self.new_departed.add(veh)

        self.new_arrived = {self.vehicles[veh_id] for veh_id in sim_res.arrived_vehicles_ids}
        # collided vehicles in the current simulation step
        self.new_collided = {self.vehicles[veh_id] for veh_id in sim_res.colliding_vehicles_ids}
        self.new_arrived -= self.new_collided # Don't count collided vehicles as "arrived"
        num_collision = len(self.new_collided)
        throughput_num = len(self.new_arrived)
        wandb.log({"collided_num": num_collision})
        if len(self.vehicles) != 0:
            wandb.log({"collided_rate": num_collision / len(self.vehicles)})
            wandb.log({"throughput_rate": throughput_num / len(self.vehicles)})
        else:
            wandb.log({"collided_rate": 0})
            wandb.log({"throughput_rate": 0})
        wandb.log({"throughput": throughput_num})
        self.collision_num.append(num_collision)
        self.junction_throughput.append(throughput_num)

        # remove arrived vehicles from Env
        for veh in self.new_arrived:
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)
            self.vehicles.pop(veh.id)

        # update vehicles' info for Env
        removed_vehicle = []
        for veh_id, veh in self.vehicles.items():
            veh.prev_speed = veh.get('speed', None)
            is_empty, namespace = self.sumo_interface.subscribes.veh.get(veh_id)
            if not is_empty:
                veh.update(namespace)
                if veh.type == 'RL':
                    self.rl_vehicles[veh_id].update(namespace)
            else:
                removed_vehicle.append(veh)
        for veh in removed_vehicle:
            self.vehicles.pop(veh.id)
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)
        ## update obs 
        obs, rewards, dones = self._update_obs(action)
        infos = {}
        truncated = False
        if self._step >= self._max_episode_steps:
            truncated = True
            dones = True
            #self.close()
        self._step += 1
        self.previous_obs, self.previous_dones\
              = deepcopy(obs), deepcopy(dones)
        return obs, rewards, dones, truncated, infos
    
    
    def reset(self, *, seed=None, options=None):
        if self.wandb_id != "":
            wandb.init(
                project = "Traffic",
                group = "roundabout",
                id = self.wandb_id,
                resume = "must"
        )
        else:
            if self.wandb_name != "":
                wandb.init(
                    project = "Traffic",
                    group = "roundabout",
                    name = self.wandb_name
                )
            else:
                wandb.init(
                    project = "Traffic",
                    group = "roundabout",
                    name = "ra_baseline_training"
                )
        self._print_debug('reset')
        # soft reset
        while not self.sumo_interface.reset_sumo():
            pass
        # if self.rl_prob_list:
        #     self.default_rl_prob = random.choice(self.rl_prob_list)
        #     print("new RV percentage = "+str(self.default_rl_prob))
        self.init_env()
        obs = []
        if options:
            if options['mode'] == 'HARD':
                obs, _, _, _, infos = self.step_once()
                return obs, infos
        while len(obs)==0:
            obs, _, _, _, infos = self.step_once()
        return obs, infos


    def step(self, action=[]):
        if len(action) == 0:
            print("empty action")
        obs, rewards, dones, truncated, infos = self.step_once(action)
        return obs, rewards, dones, truncated, infos


    def close(self):
        self.sumo_interface.close()


    def _print_debug(self, fun_str):
        if self.print_debug:
            print('exec: '+fun_str+' at time step: '+str(self._step))

    
    def virtual_id_assign(self, veh_id):
        if not veh_id in self.veh_name_mapping_table.keys():
            self.veh_name_mapping_table[veh_id] = (veh_id, False)
            return veh_id
        else:
            if self.veh_name_mapping_table[veh_id][1]:
                virtual_new_id = veh_id+'@'+str(10*random.random())
                self.veh_name_mapping_table[veh_id] = (virtual_new_id, False)
                return virtual_new_id
            else:
                return self.veh_name_mapping_table[veh_id][0]
            

    def convert_virtual_id_to_real_id(self, virtual_id):
        return virtual_id.split('@')[0]


    def terminate_veh(self, virtual_id):
        real_id = virtual_id.split('@')[0]
        self.veh_name_mapping_table[real_id] = (self.veh_name_mapping_table[real_id][0], True)


    def _traffic_light_program_update(self):
        if self._step> self.traffic_light_program['disable_light_start']:
            self.sumo_interface.disable_all_trafficlight(self.traffic_light_program['disable_state'])






























