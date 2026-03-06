#!/bin/bash
# First arg is how many seeds (including the original), second arg should be the full filename of the sb file

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_seeds> <sb_filename>"
    exit 1
fi

num_seeds=$1
base_file=$2

# Validate if the first argument is a number
if ! [[ "$num_seeds" =~ ^[0-9]+$ ]]; then
    echo "Error: The first argument must be a number."
    exit 1
fi

# Process subsequent seeds (starting from 2 as per original script logic)
for i in $(seq 2 "$num_seeds")
do
    output_file="${base_file}_s$i"

    # Copy the base file
    cp "$base_file" "$output_file"

    # Extract the original master_port from the *copied* file
    # Using grep to find the line and sed to extract the number
    original_port=$(grep "master_port" "$output_file" | sed 's/.*master_port \([0-9]*\).*/\1/')

    # Check if original_port was found and is a number
    if ! [[ "$original_port" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not find a valid master_port in $output_file. Skipping port modification for this file."
        # Decide if you want to skip the rest of the loop iteration or exit
        # continue # Use 'continue' to skip to the next seed
        # exit 1 # Use 'exit 1' to stop the script
    else
        # Calculate the new port number
        new_port=$((original_port + i - 1))

        # Use sed to substitute the entire original port number with the new one
        # We only do this if we successfully extracted the original port
        sed -i "s@master_port $original_port@master_port $new_port@" "$output_file"
    fi

    # Perform other necessary substitutions
    sed -i "s@seed 1@seed $i@" "$output_file"
    # Assuming "J s1" is preceded by something, like - or --
    # If it's just "J s1", use "s@J s1@J s$i@"
    sed -i "s@J s1@J s$i@" "$output_file"
    sed -i "s@1.txt@$i.txt@" "$output_file"

done
