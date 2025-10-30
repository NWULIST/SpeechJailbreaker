#!/bin/bash

# Function to check if run_JBC.sh is still running
is_running() {
    pgrep -f run_JBC.sh > /dev/null
    return $?
}

# Loop until run_JBC.sh is no longer running
while is_running; do
    echo "run_JBC.sh is still running. Checking again in 10 minutes..."
    sleep 120  # Check every 2 minutes
done

echo "run_JBC.sh has finished. Starting run_JBC.sh..."
./Scripts/run_JBC.sh