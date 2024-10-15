#!/bin/bash

# Function to convert single .osm file
convert_single_osm() {
    local OSM_FILE=$1
    local BASE_NAME=$(basename "$OSM_FILE" .osm)
    mkdir -p "files/outputs/${BASE_NAME}_dict/${BASE_NAME}"
    # netconvert --osm-files "$OSM_FILE" --output-file "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --default.junctions.keep-clear false
    
    netconvert -s "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --o "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --tls.rebuild --tls.default-type actuated
    #python keep_roundabout.py "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml"
    $SUMO_HOME/tools/randomTrips.py -n "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" -e 10000 -p 0.5 -o "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" --fringe-factor max --validate
    duarouter -n "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --route-files "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" -o "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.rou.xml" --ignore-errors
    duarouter -n "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.net.xml" --route-files "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.trips.xml" -o "files/outputs/${BASE_NAME}_dict/${BASE_NAME}/${BASE_NAME}.rou.alt.xml" --ignore-errors
    cat <<EOL > "files/outputs/${BASE_NAME}_dict/${BASE_NAME}.sumo.cfg"
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
convert_all_osm() {
    for file in files/osm_files/*.osm; do
        convert_single_osm "$file"
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
    convert_single_osm "$1"
fi

echo "All steps completed."
