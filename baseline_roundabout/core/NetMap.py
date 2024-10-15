import sys, os
sys.path.append(os.getcwd())
from core.utils import map_parser, detect_all_junctions
from copy import deepcopy
import math
import traci  as T
import numpy as np

class NetMap(object):
    def __init__(self, xml_path) -> None:
        self.xml_path = xml_path
        self.net_data, self.connection_data, self.junction_data, self.roundabout_data = map_parser(self.xml_path)
        self.junction_list = detect_all_junctions(self.junction_data)
        self.incoming_edges, self.junction_incoming_edges = self._compute_turning_map()
        self.incoming_edges_recursive = self._compute_recursion_incoming(recurse_step=3)
        self._keyword_check()
    
    def _keyword_check(self):
        for _, edge in self.incoming_edges.items():
            if 'straight' not in edge:
                edge['straight'] = []
            if 'left' not in edge:
                edge['left'] = []
            if 'right' not in edge:
                edge['right'] = []
        
        for _, edge in self.incoming_edges_recursive.items():
            if 'straight' not in edge:
                edge['straight'] = []
            if 'left' not in edge:
                edge['left'] = []
            if 'right' not in edge:
                edge['right'] = []
    
    def prev_edge(self, edge, lane, skip_junction=False):
        """See parent class."""
        if skip_junction:
            try:
                tlanes = self.connection_data['prev'][edge][lane]
                result = deepcopy(tlanes)
                for tlane in tlanes:
                    if tlane[0][0] == ':':
                        result.extend(self.prev_edge(tlane[0], tlane[1]))
                        result.remove(tlane)
                return result
            except KeyError:
                return []
        else:
            try:
                return self.connection_data['prev'][edge][lane]
            except KeyError:
                return []

    def next_edge(self, edge, lane, skip_junction=False):
        """See parent class."""
        if skip_junction:
            try:
                tlanes = self.connection_data['next'][edge][lane]
                result = deepcopy(tlanes)
                for tlane in tlanes:
                    if tlane[0][0] == ':':
                        result.extend(self.next_edge(tlane[0], tlane[1]))
                        result.remove(tlane)
                return result
            except KeyError:
                return []
        else:
            try:
                return self.connection_data['next'][edge][lane]
            except KeyError:
                return []

    def get_edge_veh_lanes(self, edge_id):
        lane_ids = []
        if not edge_id in self.net_data.keys():
            print('invalid edge: '+edge_id)
            return lane_ids
        for lane_id in self.net_data[edge_id]['lanes'].keys():
            if self._allow_car(edge_id, lane_id.split('_')[-1]): 
                lane_ids.extend([lane_id.split('_')[-1]])
        return lane_ids
    
    def edge_length(self, edge_id):
        try:
            return self.net_data[edge_id]['length']
        except:
            print("fail to load edge length of "+str(edge_id))
            return -1.0
    
    def junction_pos(self, junc_id):
        return [self.junction_data[junc_id]['x'], self.junction_data[junc_id]['y']]
    
    def _allow_car(self, EdgeID, LaneID):
        ## return whether the lane allows passenger cars
        return (not self.net_data[EdgeID]['lanes'][EdgeID+'_'+str(LaneID)]['type']) \
             or 'passenger' in self.net_data[EdgeID]['lanes'][EdgeID+'_'+str(LaneID)]['type'] 

    def identify_lanedir(edge_dict, edge_id):
        edge_dict["unidentified"] = list(set(edge_dict["unidentified"]))
        edge_dict["unidentified"].sort()
        previous_edge = 0
        lane_group_idx = 0
        edge_lane_group = []
        for lane_idx in edge_dict["unidentified"]:
            target_edge = self.next_edge(edge_id, lane_idx, True)
            if len(target_edge) == 0:
                continue
            if previous_edge == 0:
                previous_edge = target_edge[0][0]
                edge_lane_group.append([lane_idx])
                continue
            if previous_edge != target_edge[0][0]:
                lane_group_idx += 1
                edge_lane_group.append([lane_idx])
            else:
                edge_lane_group[lane_group_idx].extend([lane_idx])
            previous_edge = target_edge[0][0]
        return edge_lane_group

    def _compute_turning_map(self):
        incoming_edges = dict()
        junction_incoming_edges = dict()
        for junc_id, juncs in self.junction_data.items():
            if len(self.junction_list)> 0 and junc_id not in self.junction_list:
                continue
            inclanes = juncs["incLanes"]
            junction_incoming_edges[junc_id] = []
            for lane in inclanes:
                if len(lane) < 2:
                    continue
                edge_id = lane[:-2]
                lane_id = lane[-1:]
                if edge_id not in incoming_edges.keys():
                    incoming_edges[edge_id] = dict()
                    junction_incoming_edges[junc_id].extend([edge_id])
                if self._allow_car(edge_id, lane_id):
                    try:
                        incoming_edges[edge_id]["unidentified"].extend([int(lane_id)])
                    except:
                        incoming_edges[edge_id]["unidentified"] = [int(lane_id)]
                incoming_edges[edge_id]["junction"] = junc_id
        # pop invalid edges from two dicts
        pop_edge_list = []
        for edge_id, _ in incoming_edges.items():
            if "unidentified" not in incoming_edges[edge_id] or len(incoming_edges[edge_id]["unidentified"]) < 2:
                pop_edge_list.extend([edge_id])
        for edge_to_pop in pop_edge_list:
            incoming_edges.pop(edge_to_pop)
            for _, edge_list in junction_incoming_edges.items():
                try:
                    edge_list.remove(edge_to_pop)
                except:
                    pass
        # may need to add sth here; previously: add NE
                
        # compute lane direction without using fixed label
        for edge_id, edge_dict in incoming_edges.items():
            edge_dict['unidentified'] = list(set(edge_dict['unidentified']))
            edge_dict['unidentified'].sort()
            previous_edge = 0
            lane_group_idx = 0
            edge_lane_group = []
            for lane_idx in edge_dict['unidentified']:
                target_edge = self.next_edge(edge_id, lane_idx, True)
                if len(target_edge) == 0:
                    continue
                if not previous_edge:
                    previous_edge = target_edge[0][0]
                    edge_lane_group.append([lane_idx])
                    continue
                if previous_edge != target_edge[0][0]:
                    lane_group_idx += 1
                    edge_lane_group.append([lane_idx])
                else:
                    edge_lane_group[lane_group_idx].extend([lane_idx])
                previous_edge = target_edge[0][0]
            
            # Label lanes dynamically
            edge_dict["group"] = edge_lane_group
        return incoming_edges, junction_incoming_edges
  
    def _compute_recursion_incoming(self, recurse_step):
        incoming_edges_recursive = deepcopy(self.incoming_edges)
        for root_edge_id in self.incoming_edges.keys():
            current_edge = [root_edge_id]
            JuncID = self.get_facing_intersection(root_edge_id, False)
            for _ in range(recurse_step):
                new_current = []
                for edge_id in current_edge:
                    prev_edges, incoming_edges_recursive = self._add_recurse_intersection_edge(edge_id, JuncID, incoming_edges_recursive)
                    new_current.extend(prev_edges)
                current_edge = deepcopy(new_current)
        return incoming_edges_recursive
    
    def _add_recurse_intersection_edge(self, edge_id, junc_id, recurse_edge_list):
        results = []
        prev_edges = []
        for lane_id in range(self.net_data[edge_id]["numlane"]):
            prev_edge_list = self.prev_edge(edge_id, lane_id)
            results.extend(prev_edge_list)
        for r in results:
            if not self.net_data[r[0]]['lanes'][r[0]+'_'+str(r[1])]["type"]:
                if r[0] not in prev_edges:
                    prev_edges.extend([r[0]])
                if r[0] not in recurse_edge_list.keys():
                    recurse_edge_list[r[0]] = dict()
                if "unidentified" not in recurse_edge_list[r[0]].keys():
                    recurse_edge_list[r[0]]["unidentified"] = [r[1]]
                else:
                    recurse_edge_list[r[0]]["unidentified"].extend([r[1]])
                recurse_edge_list[r[0]]["junction"] = junc_id
        return prev_edges, recurse_edge_list

    def get_distance_to_intersection(self, veh):
        junc_id = self.get_facing_intersection(veh.road_id)
        if len(junc_id) == 0:
            return 1000000
        junc_pos = self.junction_pos(junc_id)
        return math.sqrt((veh.position[0]-float(junc_pos[0]))**2+(veh.position[1]-float(junc_pos[1]))**2)
    
    def get_facing_intersection(self, edge_id, recursion=True):
        try:
            if recursion:
                return self.incoming_edges_recursive[edge_id]['junction']
            else:
                return self.incoming_edges[edge_id]['junction']
        except KeyError:
            return []
    
    def get_veh_moving_direction(self, veh):
        # if not an inner road
        if veh.road_id[0] != ':':
            facing_junction_id = self.get_facing_intersection(veh.road_id)
        else:
            for ind in range(len(veh.road_id)):
                if veh.road_id[len(veh.road_id)-1-ind] == '_':
                    break
            last_dash_ind = len(veh.road_id)-1-ind
            facing_junction_id = veh.road_id[1:last_dash_ind]
        return facing_junction_id, T.vehicle.getLaneID(veh.id)

    
    def get_junction_throughput_in_step(self, junction_id):
        """
        Get the throughput of a specified junction in the current simulation step.
        
        Parameters:
        junction_id: str, ID of the junction
        
        Returns:
        throughput: int, number of vehicles passing through the junction in the current simulation step
        """
        # Get all lanes connected to the junction
        all_lanes = T.junction.getLanes(junction_id)

        # Initialize the counter
        throughput = 0

        # Iterate through each lane and count the passing vehicles
        for lane_id in all_lanes:
            edge_id = T.lane.getEdgeID(lane_id)
            if edge_id.startswith(":"):  # Filter out internal edges
                continue
            throughput += len(T.edge.getLastStepVehicleIDs(edge_id))

        return throughput

    # def get_junction_throughput_in_step(self, junction_id):
    #     """
    #     Get the throughput of a specified junction in the current simulation step.
        
    #     Parameters:
    #     junction_id: str, ID of the junction
        
    #     Returns:
    #     throughput: int, number of vehicles passing through the junction in the current simulation step
    #     """
    #     # Get all edges connected to the junction
    #     all_edges = T.junction.getEdges(junction_id)

    #     # Filter out the outgoing edges
    #     outgoing_edges = [edge_id for edge_id in all_edges if T.lane.getEdgeID(T.lane.getLastLane(edge_id)) != junction_id]

    #     # Initialize the counter
    #     throughput = 0

    #     # Iterate through each outgoing edge and count the passing vehicles
    #     for edge_id in outgoing_edges:
    #         throughput += len(T.edge.getLastStepVehicleIDs(edge_id))

    #     return throughput
    
    # def get_junction_throughput_in_step(self, junction_id):
    #     """
    #     Get the throughput of a specified junction in the current simulation step.
        
    #     Parameters:
    #     junction_id: str, ID of the junction
        
    #     Returns:
    #     throughput: int, number of vehicles passing through the junction in the current simulation step
    #     """
    #     # Get the outgoing edges of the junction
    #     outgoing_edges = T.junction.getOutgoing(junction_id)
    #     outgoing_edge_ids = [edge['id'] for edge in outgoing_edges]

    #     # Initialize the counter
    #     throughput = 0

    #     # Iterate through each outgoing edge and count the passing vehicles
    #     for edge_id in outgoing_edge_ids:
    #         throughput += len(T.edge.getLastStepVehicleIDs(edge_id))

    #     return throughput
    















































