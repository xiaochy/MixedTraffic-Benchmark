#!/bin/bash

ROUNDABOUT_PATH="train_files/roundabout/middle"
INTERSECTION_PATH="train_files/intersection/middle"
# Function to convert single .osm file
convert_single_osm() {
    # local ROUNDABOUT_PATH="benchmark_files/roundabout/easy"
    # local INTERSECTION_PATH="benchmark_files/intersection/easy"
    local OSM_FILE=$1
    local SCENARIO=$2
    local MYPATH=""
    local BASE_NAME=$(basename "$OSM_FILE" .osm) # input: benchmark_files/roundabout/easy/osm_files/xxx.osm  BASE_NAME=xxx
    if [ "$SCENARIO" = "roundabout" ]; then
        local MYPATH="$ROUNDABOUT_PATH"
    elif [ "$SCENARIO" = "intersection" ]; then
        local MYPATH="$INTERSECTION_PATH"
    fi
    #echo "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}"
    mkdir -p "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}"
    netconvert --osm-files "$OSM_FILE" --output-file "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --default.junctions.keep-clear false
    #python keep_roundabout.py "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml"
    $SUMO_HOME/tools/randomTrips.py -n "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" -p 1.2 -o "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" --fringe-factor max --validate
    duarouter -n "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --route-files "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" -o "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.rou.xml" --ignore-errors
    duarouter -n "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --route-files "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" -o "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.rou.alt.xml" --ignore-errors
    cat <<EOL > "$MYPATH/outputs/${BASE_NAME}_dict/${BASE_NAME}.sumo.cfg"
<?xml version="1.0" encoding="iso-8859-1"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/duarouterConfiguration.xsd">

    <input>
        <net-file value="${BASE_NAME}/${BASE_NAME}.net.xml"/>
        <route-files value="${BASE_NAME}/${BASE_NAME}.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="86400"/>
    </time>

</configuration>
EOL
    echo "Converted $OSM_FILE"
}

# Function to convert all .osm files in osm_files directory
# convert_all_osm() {
#     for file in benchmark_files/roundabout/easy/osm_files/*.osm; do
#         convert_single_osm "$file"
#     done
#     for file in benchmark_files/intersection/easy/osm_files/*.osm; do
#         convert_single_osm "$file"
#     done
# }

convert_all_osm() {
    for file in $ROUNDABOUT_PATH/osm_files/*.osm; do
        # 提取路径中的 roundabout 或 intersection 部分
        #SCENARIO=$(echo "$file" | awk -F'/' '{print $3}')
        
        # 传递 file 和 SCENARIO 给 convert_single_osm 函数
        convert_single_osm "$file" "roundabout"
    done
    for file in $INTERSECTION_PATH/osm_files/*.osm; do
        # 提取路径中的 roundabout 或 intersection 部分
        #SCENARIO=$(echo "$file" | awk -F'/' '{print $3}')
        
        # 传递 file 和 SCENARIO 给 convert_single_osm 函数
        convert_single_osm "$file" "intersection"
    done
}


# Check if an .osm file or the '-a' option is provided as an argument
if [[ $# -eq 0 || "$1" != "-a" && ! -f "$1" ]]; then
    echo "Usage:"
    echo "To convert a single .osm file: $0 <path_to_osm_file>"
    echo "To convert all .osm files in files/osm_files directory: $0 -a"
    exit 1
fi

# Convert either a single .osm file or all .osm files in osm_files directory based on the provided argument
if [ "$1" == "-a" ]; then
    convert_all_osm
else
    convert_single_osm "$1" "$2"
fi

echo "All steps completed."
