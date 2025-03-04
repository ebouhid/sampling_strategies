import argparse
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def run_command(cmd, out_file_path):
    """Helper function to run a command and capture output"""
    with open(out_file_path, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    return (cmd, result.returncode)

def main():
    parser = argparse.ArgumentParser(description='Run parallel curriculum learning experiments')
    parser.add_argument('--exp_name', required=True, help='Experiment name for results directory')
    parser.add_argument('--num_workers', "-n", type=int, default=4, 
                      help='Number of parallel workers (default: 4)')
    args = parser.parse_args()
    
    output_dir = f"results_{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all commands first
    commands = []
    datasets = ['lsat_8', 'sentinel']
    criteria = ['dtb', 'entropy', 'time']
    
    for dataset in datasets:
        for criterion in criteria:
            segs_csv = os.path.join('dataset_har', dataset, f"{criterion}.csv")
            segs_dir = os.path.join('dataset_har', dataset)
            output_csv = os.path.join(output_dir, f"{dataset}_{criterion}.csv")
            out_file_path = os.path.join(output_dir, f"{dataset}_{criterion}.out")
            ckpt_path = os.path.join(output_dir, "model_checkpoints")
            
            cmd = [
                'python', 'svm_curriculum.py',
                '--segs_csv', segs_csv,
                '--segs_dir', segs_dir,
                '--output', output_csv,
                '--exp_name', args.exp_name,
                '--checkpoint_dir', ckpt_path
            ]
            
            commands.append((cmd, out_file_path))
    
    # Run commands in parallel
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for cmd, out_file_path in commands:
            futures.append(executor.submit(run_command, cmd, out_file_path))
        
        # Check for errors
        error_count = 0
        for future in futures:
            (cmd, returncode) = future.result()
            if returncode != 0:
                error_count += 1
                print(f"Error in command: {' '.join(cmd)}")
                print(f"Exit code: {returncode}")
    
    print(f"\nDone! Completed {len(commands)} tasks with {error_count} errors.")

if __name__ == "__main__":
    main()