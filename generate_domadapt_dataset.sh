python_script="generate_domadapt_data.py"

# Function to handle build_domadapt_dataset_outs directory
handle_output_directory() {
    output_dir="build_domadapt_dataset_outs"
    if [ -d "$output_dir" ]; then
        echo "Output directory $output_dir exists, clearing files"
        rm -f "$output_dir"/*.out
    else
        echo "Creating output directory $output_dir"
        mkdir "$output_dir"
    fi
}

# Run the function to handle the output directory
handle_output_directory

# Define the regions
regions=("x01" "x02" "x03" "x04" "x06" "x07" "x08" "x09" "x10")

# Loop through regions, excluding x05, and run Python script with nohup
for region in "${regions[@]}"; do
    if [ "$region" != "x05" ]; then
        output_file="build_domadapt_dataset_outs/$region.out"
        echo "Running $python_script for region $region with nohup, output saved to $output_file"
        nohup python "$python_script" "$region" > "$output_file" 2>&1 &
    fi
done

echo "All processes started in the background. Check individual output files in build_domadapt_dataset_outs directory."
