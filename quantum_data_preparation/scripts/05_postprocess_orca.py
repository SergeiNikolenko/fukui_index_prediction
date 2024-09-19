# Post-processes ORCA results, preparing them for further analysis.

import os
import shutil
import subprocess
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

# Set paths and logging
base_dir = Path(".")
orca_path = "/home/nikolenko/work/orca/orca/orca_2mkl"
log_file = base_dir / "process.log"


def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")


# Function to process ORCA output files with JANPA
def prepare_and_process_file(out_file):
    base_name = out_file.stem
    tmp_dir = base_dir / "tmp" / base_name

    # Skip already processed files
    if (base_dir / f"janpa_done/{base_name}.out").exists():
        log(f"File {base_name} already processed, skipping.")
        return False

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Copy necessary files
    log(f"Copying files for {base_name}")
    shutil.copy(base_dir / f"orca_calc/{base_name}.out", tmp_dir)
    shutil.copy(base_dir / f"orca_calc/{base_name}.gbw", tmp_dir)

    # Run ORCA's molden conversion
    log(f"Running ORCA for {base_name}")
    subprocess.run(
        f"{orca_path} {base_name} -molden",
        shell=True,
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Rename molden input file
    log(f"Renaming molden.input to molden for {base_name}")
    subprocess.run(
        f"mv {base_name}.molden.input {base_name}.molden",
        shell=True,
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Run molden2molden conversion
    log(f"Running molden2molden for {base_name}")
    subprocess.run(
        f"java -jar ../../janpa/molden2molden.jar -orca3signs -fromorca3bf -i {base_name}.molden -o {base_name}_ok.molden",
        shell=True,
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Run JANPA for population analysis
    log(f"Running JANPA for {base_name}")
    subprocess.run(
        f"java -jar ../../janpa/janpa.jar {base_name}_ok.molden > {base_name}_janpa.out",
        shell=True,
        cwd=tmp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Move JANPA output to final directory
    log(f"Moving {base_name}_janpa.out to janpa_done")
    (tmp_dir / f"{base_name}_janpa.out").rename(
        base_dir / f"janpa_done/{base_name}.out"
    )

    # Cleanup temporary files
    log(f"Deleting temporary directory for {base_name}")
    shutil.rmtree(tmp_dir)

    return True


# Get list of ORCA output files
out_files = list(base_dir.glob("orca_calc/*.out"))
completed_files = [
    f for f in out_files if (base_dir / f"janpa_done/{f.stem}.out").exists()
]
remaining_files = [f for f in out_files if f not in completed_files]

log(
    f"Total files: {len(out_files)}, Completed: {len(completed_files)}, Remaining: {len(remaining_files)}"
)

# Create 'janpa_done' directory if it doesn't exist
(base_dir / "janpa_done").mkdir(exist_ok=True)

# Process each remaining ORCA file using parallel processing
results = Parallel(n_jobs=56)(
    delayed(prepare_and_process_file)(out_file)
    for out_file in tqdm(remaining_files, desc="Processing files")
)
results = [res for res in results if res]

log("Done!")
