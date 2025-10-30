#!/bin/bash

# Function to check if run_PAIR.sh is still running
is_running() {
    pgrep -f run_PAIR.sh > /dev/null
    return $?
}

# Loop until run_PAIR.sh is no longer running
while is_running; do
    echo "run_PAIR.sh is still running. Checking again in 10 minutes..."
    sleep 120  # Check every 2 minutes
done

echo "run_PAIR.sh has finished. Starting run_PAIR.sh..."
./Scripts/run_PAIR.sh