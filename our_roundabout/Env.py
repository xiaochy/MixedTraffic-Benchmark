from typing import Set
import random, math
from copy import deepcopy
import numpy as np
import wandb
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



class Env(MultiAgentEnv):
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
        self.rl_prob_list = config['rl_prob_range'] if 'rl_prob_range' in config.keys() else None
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
        self.n_obs = (FRONT_VEH_NUM + BACK_VEH_NUM)*4
        self.action_space = Box(low=self.min_acc, high=self.max_acc, shape=(1,),dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(self.n_obs,),dtype=np.float32)
        self.veh_count = 0
        self.agressive_veh = []
        self.conservative_veh = []
        self.wandb_id = config["wandb_id"]
        if 'wandb_name' in config.keys():
            self.wandb_name = config["wandb_name"]
        #self.wandb_name = config["wandb_name"]
        #self.track_veh = []
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
        self._print_debug('init_env')


    def _update_obs(self,action, reward):
        obs = {}
        rewards = {}
        dones = {}
        avg_waiting = 0
        position_dict = {}
        velocity_dict = {}
        angle_radian_dict = {}
        direction_dict = {}

        for veh in self.vehicles:
            # check wait_time
            wait_time, accum_wait_time = self.sumo_interface.get_veh_waiting_time(veh)
            avg_waiting += wait_time
            position_dict[veh.id] = self.sumo_interface.get_position(veh)
            velocity_dict[veh.id], angle_radian_dict[veh.id], direction_dict[veh.id] = self.sumo_interface.get_velocity_angle_direction(veh)
        if len(self.vehicles) != 0:
            avg_waiting /= len(self.vehicles)
    
        wandb.log({"average_waiting_time": avg_waiting})
        # check self.avg_waiting_time
        self.avg_waiting_time.append(avg_waiting)
        reward_new = reward + self.gamma*self.wait_time_reward(avg_waiting)
        wandb.log({"reward": reward_new})
    

        for rl_veh in self.rl_vehicles:
            virtual_id = self.virtual_id_assign(rl_veh.id) 
            if len(rl_veh.road_id) == 0:
                if virtual_id in action.keys():
                    obs[virtual_id] = self.check_obs_constraint(self.previous_obs[virtual_id])
                    rewards[virtual_id] = reward
                    dones[virtual_id] = True
                    self.terminate_veh(virtual_id)
                    continue
                else:
                    continue
            obs_relative_position = []
            obs_relative_velocity = []
            obs_relative_position_front = []
            obs_relative_velocity_front = []
            obs_relative_position_back = []
            obs_relative_velocity_back = []
            position = position_dict[rl_veh.id]
            velocity, angle_radian, direction = velocity_dict[rl_veh.id], angle_radian_dict[rl_veh.id], direction_dict[rl_veh.id]
            front_center = position + (FRONT_DISTANCE / 2) * direction
            back_center = position - (BACK_DISTANCE / 2) * direction
            front_cnt, back_cnt = 0, 0 
            # sort the vehicles according to the distance to the center car from near to far
            vehicles_with_distances = [(veh_id, self.sumo_interface.calculate_distance(veh_pos, position)) for veh_id, veh_pos in position_dict.items()]
            vehicles_with_distances.sort(key=lambda x: x[1])
            # Traverse the sorted_vehicles list and update obs
            for veh_id, _ in vehicles_with_distances:
                # Don't add its own info to its obs
                if veh_id == rl_veh.id:
                    continue
                neigh_pos, neigh_vel = position_dict[veh_id], velocity_dict[veh_id]
                relative_pos, relative_vel = neigh_pos - position, neigh_vel - velocity
                relative_pos = self.rotate_vector(relative_pos, angle_radian)
                x = relative_pos[0]
                y = relative_pos[1]
                if front_cnt < FRONT_VEH_NUM:
                    if self.is_point_in_rotated_rectangle(neigh_pos[0], neigh_pos[1], front_center[0], front_center[1], FRONT_DISTANCE, 2*WIDTH, angle_radian):
                        front_cnt += 1
                        obs_relative_position_front.extend(relative_pos.tolist())
                        obs_relative_velocity_front.extend(relative_vel.tolist())
                if back_cnt < BACK_VEH_NUM:
                    if self.is_point_in_rotated_rectangle(neigh_pos[0], neigh_pos[1], back_center[0], back_center[1], BACK_DISTANCE, 2*WIDTH, angle_radian):
                        back_cnt += 1
                        obs_relative_position_back.extend(relative_pos.tolist())
                        obs_relative_velocity_back.extend(relative_vel.tolist())
                if front_cnt == FRONT_VEH_NUM and back_cnt == BACK_VEH_NUM:
                    break
            # normalize obs_relative_position_front to x in [-1,1], y in [0,1]
            for idx in range(len(obs_relative_position_front)):
                if idx % 2 == 0:
                    obs_relative_position_front[idx] /= WIDTH
                else:
                    obs_relative_position_front[idx] /= FRONT_DISTANCE

            # normalize obs_relative_position_back to x in [-1,1], y in [-1,0]
            for idx in range(len(obs_relative_position_back)):
                if idx % 2 == 0:
                    obs_relative_position_back[idx] /= WIDTH
                else:
                    obs_relative_position_back[idx] /= BACK_DISTANCE
            # record the max speed along the velocity axis x and y
            for idx in range(len(obs_relative_velocity_front)):
                if abs(obs_relative_velocity_front[idx]) > MAX_FRONT_SPEED:
                    if obs_relative_velocity_front[idx] < 0:
                        obs_relative_velocity_front[idx] = -MAX_FRONT_SPEED
                    else:
                        obs_relative_velocity_front[idx] = MAX_FRONT_SPEED
                obs_relative_velocity_front[idx] /= MAX_FRONT_SPEED

            for idx in range(len(obs_relative_velocity_back)):
                if abs(obs_relative_velocity_back[idx]) > MAX_BACK_SPEED:
                    if obs_relative_velocity_back[idx] < 0:
                        obs_relative_velocity_back[idx] = -MAX_BACK_SPEED
                    else:
                        obs_relative_velocity_back[idx] = MAX_BACK_SPEED
                obs_relative_velocity_back[idx] /= MAX_BACK_SPEED    
             
            front_remain = FRONT_VEH_NUM - front_cnt
            for _ in range(front_remain):
                obs_relative_position_front.extend([1, 1])
                obs_relative_velocity_front.extend([0, 0])
            back_remain = BACK_VEH_NUM - back_cnt
            for _ in range(back_remain):
                obs_relative_position_back.extend([-1, -1])
                obs_relative_velocity_back.extend([0, 0])
            
            obs_relative_position = obs_relative_position_front + obs_relative_position_back
            obs_relative_velocity = obs_relative_velocity_front + obs_relative_velocity_back
            obs_relative_position, obs_relative_velocity = np.array(obs_relative_position), np.array(obs_relative_velocity)
            # set the observation of this rv ->  shape = (FRONT_VEH_NUM + BACK_VEH_NUM)*4
            obs[virtual_id] = self.check_obs_constraint(np.concatenate([obs_relative_position, obs_relative_velocity])) 
            rewards[virtual_id] = reward + self.gamma * self.wait_time_reward(self.avg_waiting_time[-1])
            dones[virtual_id] = False
        return obs, rewards, dones
            
        
    def step_once(self, action={}):
        self.new_departed = set()
        self.sumo_interface.set_max_speed_all(self.max_speed)
        self._traffic_light_program_update()
        if not (isinstance(action, dict) and len(action) == len(self.previous_obs)- sum(dict_tolist(self.previous_dones))):
            print('error!! action dict is invalid')
            return dict()
        for veh_id, _ in self.vehicles.items():
            if veh_id not in self.rl_vehicles:
                if random.random() < 0.5:
                    self.sumo_interface.accl_control(self.vehicles[veh_id], self.max_acc)
                else:
                    self.sumo_interface.accl_control(self.vehicles[veh_id], self.min_acc)
        # for veh_id in self.vehicles:
        #     if veh_id not in self.rl_vehicles:
        #         if random.random() < 0.5:
        #             self.sumo_interface.accl_control(self.vehicles[veh_id], self.max_acc)
        #         else:
        #             self.sumo_interface.accl_control(self.vehicles[veh_id], self.min_acc)
        # for veh_id in self.agressive_veh:
        #     if veh_id in self.vehicles:
        #         if random.random() < 0.5:
        #             self.sumo_interface.accl_control(self.vehicles[veh_id], self.max_acc)
        # for veh_id in self.conservative_veh:
        #     if veh_id in self.vehicles:
        #         if random.random() < 0.5:
        #             self.sumo_interface.accl_control(self.vehicles[veh_id], self.min_acc)
            
        # apply action on each rl vehicle
        for virtual_id in action.keys():
            veh_id = self.convert_virtual_id_to_real_id(virtual_id)
            acc = action[virtual_id]
            self.sumo_interface.accl_control(self.rl_vehicles[veh_id], acc[0])

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
                #T.vehicle.setLaneChangeMode(veh_id,0)
                self.sumo_interface.tc.vehicle.setSpeedMode(veh_id,39)
            else:
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="IDM", route=route, length=length, wait_time=0)
                # if random.random() < 0.5:
                #     self.conservative_veh.append(veh_id)
                # else:
                #     self.agressive_veh.append(veh_id)
                self.sumo_interface.tc.vehicle.setSpeedMode(veh_id,39)
                #T.vehicle.setSpeedMode(veh_id,0)
            self.sumo_interface.set_color(veh, WHITE if veh.type=="IDM" else RED)
            # self.veh_count += 1
            # if self.veh_count == 1:
            #     self.track_veh.append(veh_id)
            #     print("track_veh = ", self.track_veh)
            self.new_departed.add(veh)

        self.new_arrived = {self.vehicles[veh_id] for veh_id in sim_res.arrived_vehicles_ids}
        # collided vehicles in the current simulation step
        self.new_collided = {self.vehicles[veh_id] for veh_id in sim_res.colliding_vehicles_ids}
        self.new_arrived -= self.new_collided # Don't count collided vehicles as "arrived"
        wandb.log({"collided_num": len(sim_res.colliding_vehicles_ids)})
        if len(self.vehicles) != 0:
            wandb.log({"collided_rate": len(sim_res.colliding_vehicles_ids) / len(self.vehicles)})
            wandb.log({"throughput_rate": len(sim_res.arrived_vehicles_ids) / len(self.vehicles)})
        else:
            wandb.log({"collided_rate": 0})
            wandb.log({"throughput_rate": 0})
        wandb.log({"throughput": len(sim_res.arrived_vehicles_ids)})
        self.collision_num.append(len(self.new_collided))
        self.junction_throughput.append(len(self.new_arrived))

        # remove arrived vehicles from Env
        for veh in self.new_arrived:
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)
            self.vehicles.pop(veh.id)
            # if veh.id in self.track_veh:
            #     self.track_veh.remove(veh.id)

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
            # if veh.id in self.agressive_veh:
            #     self.agressive_veh.remove(veh.id)
            # if veh.id in self.conservative_veh:
            #     self.conservative_veh.remove(veh.id)
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)

        ## update obs 
        reward = self.reward_compute_global()
        obs, rewards, dones = self._update_obs(action, reward)
        dones['__all__'] = False
        infos = {}
        truncated = {}
        truncated['__all__'] = False
        if self._step >= self._max_episode_steps:
            for key in dones.keys():
                truncated[key] = True
        self._step += 1
        self.previous_obs, self.previous_dones\
              = deepcopy(obs), deepcopy(dones)
        return obs, rewards, dones, truncated, infos
    
    def reward_compute_global(self):
        reward = self.alpha * self.throughput_reward(self.junction_throughput[-1]) + self.beta * self.collision_reward(self.collision_num[-1])
        self.reward_record.append(reward)
        return reward
    
    def reset(self, *, seed=None, options=None):
        if self.wandb_id != "":
            wandb.init(
                project = "Traffic",
                group = "roundabout",
                resume = "must",
                id = self.wandb_id
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
                    name = "ra_our_training"
                )
            # if self.wandb_name != "":
            #     wandb.init(
            #         project = "Traffic",
            #         group = "roundabout",
            #         name = self.wandb_name
            #     )
            # else:
            #     wandb.init(
            #         project = "Traffic",
            #         group = "roundabout",
            #         name = "try_name"
            #     )
            
        
        self._print_debug('reset')
        # soft reset
        while not self.sumo_interface.reset_sumo():
            pass

        if self.rl_prob_list:
            self.default_rl_prob = random.choice(self.rl_prob_list)
            print("new RV percentage = "+str(self.default_rl_prob))

        self.init_env()
        obs = {}
        if options:
            if options['mode'] == 'HARD':
                obs, _, _, _, infos = self.step_once()
                return obs, infos

        while len(obs)==0:
            obs, _, _, _, infos = self.step_once()
        return obs, infos

    def step(self, action={}):
        #print(action)
        if len(action) == 0:
            print("empty action")
        obs, rewards, dones, truncated, infos = self.step_once(action)

        return obs, rewards, dones, truncated, infos

    def close(self):
        ## close env
        self.sumo_interface.close()


    def check_obs_constraint(self, obs):
        if not self.observation_space.contains(obs):
            obs= np.asarray([x if x>= self.observation_space.low[0] else self.observation_space.low[0] for x in obs]\
                                    , dtype=self.observation_space.dtype)
            obs = np.asarray([x if x<= self.observation_space.high[0] else self.observation_space.high[0] for x in obs]\
                                    , dtype=self.observation_space.dtype)
            if not self.observation_space.contains(obs):
                print('dddd')
                raise ValueError(
                    "Observation is invalid, got {}".format(obs)
                )
        return obs


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






























