import time 
import xml.etree.ElementTree as ElementTree
from lxml import etree

def dict_tolist(dict_input): 
    result = []
    if isinstance(dict_input, list):
        return dict_input
    for _, s in dict_input.items():
        result.append(s)
    return result

def map_parser(xml_path):
    parser = etree.XMLParser(recover=True)
    net_path = xml_path
    tree = ElementTree.parse(net_path, parser=parser)
    root = tree.getroot()
    # init the edge type data
    types_data = dict()
    for typ in root.findall('type'):
        type_id = typ.attrib['id']
        types_data[type_id] = dict()
        if 'speed' in typ.attrib:
            types_data[type_id]['speed'] = float(typ.attrib['speed'])
        else:
            types_data[type_id]['speed'] = None
        if 'numLanes' in typ.attrib:
            types_data[type_id]['numLanes'] = int(typ.attrib['numLanes'])
        else:
            types_data[type_id]['numLanes'] = None
    # init storage of data
    net_data = dict()
    next_conn_data = dict()
    prev_conn_data = dict() 
    junction_data = dict() 
    roundabout_data = dict()

    # collect info on the edge
    for edge in root.findall('edge'):
        edge_id = edge.attrib['id']
        net_data[edge_id] = dict()
        # set "speed" of the edge
        if 'speed' in edge:
            net_data[edge_id]['speed'] = float(edge.attrib['speed'])
        else:
            net_data[edge_id]['speed'] = None
        if 'type' in edge.attrib and edge.attrib['type'] in types_data:
            if net_data[edge_id]['speed'] is None:
                net_data[edge_id]['speed'] = float(types_data[edge.attrib['type']]['speed'])

        # set "numlane" of the edge
        net_data[edge_id]['numlane'] = 0
        for i, lane in enumerate(edge):
            net_data[edge_id]['numlane'] += 1
            if i == 0:
                net_data[edge_id]['length'] = float(lane.attrib['length'])
                if net_data[edge_id]['speed'] is None and 'speed' in lane.attrib:
                    net_data[edge_id]['speed'] = float(lane.attrib['speed'])
        if net_data[edge_id]['speed'] is None:
            net_data[edge_id]['speed'] = 30

        # set "shape" of the edge
        if 'shape' in edge.attrib:
            net_data[edge_id]['shape'] = edge.attrib['shape']
        else:
            net_data[edge_id]['shape'] = edge[0].attrib['shape']
         
        # set "lane" of the edge
        net_data[edge_id]['lanes'] = dict()
        for lane in edge.findall('lane'):
            lane_id = lane.attrib['id']
            net_data[edge_id]['lanes'][lane_id] = dict()
            if 'allow' in lane.attrib:
                net_data[edge_id]['lanes'][lane_id]['type'] = lane.attrib['allow']
            else:
                net_data[edge_id]['lanes'][lane_id]['type'] = None
    
    # collect info on the connection
    for connection in root.findall('connection'):
        from_edge = connection.attrib['from']
        from_lane = int(connection.attrib['fromLane'])
        # if from_edge is not an internal edge
        if from_edge[0] != ":":
            try:
                via = connection.attrib['via'].rsplit('_', 1)
                to_edge = via[0]
                to_lane = int(via[1])
            except:
                to_edge = connection.attrib['to']
                to_lane = int(connection.attrib['toLane'])
        else:
            to_edge = connection.attrib['to']
            to_lane = int(connection.attrib['toLane'])

        if from_edge not in next_conn_data:
            next_conn_data[from_edge] = dict()
        if from_lane not in next_conn_data[from_edge]:
            next_conn_data[from_edge][from_lane] = list()
        if to_edge not in prev_conn_data:
            prev_conn_data[to_edge] = dict()
        if to_lane not in prev_conn_data[to_edge]:
            prev_conn_data[to_edge][to_lane] = list()
        next_conn_data[from_edge][from_lane].append((to_edge, to_lane))
        prev_conn_data[to_edge][to_lane].append((from_edge, from_lane))
    connection_data = {'next': next_conn_data, 'prev': prev_conn_data}

    # collect info on the junction (all the junctions)
    for junction in root.findall('junction'):
        junction_id = junction.attrib['id']
        inclanes = junction.attrib['incLanes'].split(' ')
        intlanes = junction.attrib['intLanes'].split(' ')
        junction_data[junction_id] = dict()
        junction_data[junction_id]['incLanes'] = inclanes
        junction_data[junction_id]['intLanes'] = intlanes
        junction_data[junction_id]['x'] = junction.attrib['x']
        junction_data[junction_id]['y'] = junction.attrib['y']
        incedge = []
        for lid in inclanes:
            incedge.extend([lid[:-2]])
        junction_data[junction_id]['incEdges'] = list(set(incedge))

    # collect info on the roundabout
    roundabout_id = 0
    for roundabout in root.findall("roundabout"):
        junctions = roundabout.attrib["nodes"].split(' ')
        external_edges = roundabout.attrib["edges"].split(' ')
        roundabout_data[roundabout_id] = dict()
        roundabout_data[roundabout_id]['junctions'] = junctions
        roundabout_data[roundabout_id]['external_edges'] = external_edges
        roundabout_id += 1

    return net_data, connection_data, junction_data, roundabout_data


def detect_all_junctions(junction_data):
    #_, _, junction_data, _ = map_parser(map_xml)
    junction_list = []
    for JuncID, juncs in junction_data.items():
        inclanes = juncs['incLanes']
        incedges = []
        for lane in inclanes:
            if len(lane) < 2 or lane[0] == ":":
                continue
            if lane[:-2] not in incedges:
                incedges.extend([lane[:-2]])
        if len(incedges) > 2:
            junction_list.extend([JuncID])
    return junction_list