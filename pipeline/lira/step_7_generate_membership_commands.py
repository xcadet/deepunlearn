from pathlib import Path

def generate_membership_commands():
    """Generates commands for processing each unlearner's membership probabilities."""
    
    # List of unlearners to process
    unlearners = [
        "finetune",
        "naive", 
        "kgltop5",
        "kgltop6",
        "original",
        "kgltop3",
        "kgltop2",
        "kgltop4"
    ]
    
    # Create output directory for commands
    output_dir = Path("commands")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate command file
    command_file = output_dir / "membership_commands.txt"
    
    with open(command_file, "w") as f:
        for unlearner in unlearners:
            # Using python -m ensures proper module path resolution
            cmd = (f"python -m munl.lira.consolidate_predictions "
                  f"--unlearner {unlearner}")
            f.write(f"{cmd}\n")
    

if __name__ == "__main__":
    generate_membership_commands() 