#!/usr/bin/env python
# get_cDNA_fragment_stats.py

import subprocess
import os
import logging
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from pathlib import Path
import fileinput 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_cDNA_fragment_stats(bam_path: Path, output_dir: Path, 
                            num_pairs_to_sample: int, 
                            min_len: int, max_len: int,
                            burn_in_fraction: float = 0.1): # New parameter
    """
    Calculates the mean and standard deviation of the cDNA fragment length on intra-exonic read pairs from a BAM file.

    Returns:
        A tuple of (output_path, success_boolean, error_message_or_None).
    """

    output_path = output_dir / f"{bam_path.stem}_cDNA_frag_stats.json"

    num_to_skip = int(num_pairs_to_sample * burn_in_fraction)
    correct_path = os.environ['PATH']
    cmd = (
        f"export PATH='{correct_path}' && "
        f"samtools view -f 2 {bam_path} | "
        f"awk 'BEGIN{{OFS=\"\\t\"}} $6 !~ /N/ && $9 > {min_len} && $9 < {max_len} {{print $9}}' | "
        f"tail -n +{num_to_skip + 1} | "
        f"head -n {num_pairs_to_sample} | "
        f"awk '{{sum+=$1; sumsq+=$1^2}} END {{if (NR>0) {{mean=sum/NR; stdev=sqrt(sumsq/NR - mean^2 + 1e-9); print mean, stdev}} else {{print \"NA\", \"NA\"}}}}'"
    )
    
    try:
        # Execute the command pipeline
        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.PIPE, executable='/bin/bash', env=os.environ.copy())
        
        # Parse the output line (e.g., "253.40 114.30")
        mean_str, sd_str = result.strip().split()

        if mean_str == "NA":
            logging.warning(f"No valid intra-exonic pairs found for {bam_path.name} within the range [{min_len}, {max_len}]. Cannot calculate stats.")
            return output_path, False, f"No valid pairs found in range [{min_len}, {max_len}]"

        stats = {
            "sample_name": bam_path.stem,
            "bam_path": str(bam_path),
            "pairs_in_reservoir": num_pairs_to_sample,
            "min_length_filter": min_len,
            "max_length_filter": max_len,
            "mean": float(mean_str),
            "std_dev": float(sd_str)
        }

        # Write to a structured JSON file for easy parsing later
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        return output_path, True, None

    except subprocess.CalledProcessError as e:
        error_message = f"Error processing {bam_path.name}:\n  Return Code: {e.returncode}\n  Command: {cmd}\n  Stderr: {e.stderr.strip()}"
        logging.error(error_message)
        return output_path, False, error_message
    except (ValueError, IndexError) as e:
        error_message = f"Error parsing samtools/awk output for {bam_path.name}: {e}. Raw output: '{result.strip()}'"
        logging.error(error_message)
        return output_path, False, error_message


def process_bam_list(bam_list_file: str, output_dir: str, num_samples: int, num_workers: int, min_frag_len: int, max_frag_len: int, burn_in_fraction: float):
    """Process multiple BAM files in parallel with pre-flight checks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
   
    # Use the robust stdin/file input method
    input_source = bam_list_file if bam_list_file is not None else '-'
    bam_paths = [Path(line.strip()) for line in fileinput.input(files=input_source) if line.strip()]

    # Pre-flight check: ensure files exist
    valid_bams = [bam for bam in bam_paths if bam.exists()]
    if len(valid_bams) != len(bam_paths):
        logging.warning(f"Found {len(bam_paths) - len(valid_bams)} missing BAM files.")

    if not valid_bams:
        logging.error("No valid BAM files found to process. Exiting.")
        return []

    logging.info(f"Found {len(valid_bams)} BAM files. Processing with {num_workers} parallel workers.")
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(get_cDNA_fragment_stats, bam, output_path, num_samples, min_frag_len, max_frag_len, burn_in_fraction): bam 
                   for bam in valid_bams}
        
        for future in as_completed(futures):
            bam_path = futures[future]
            out_path, success, error = future.result()
            if success:
                logging.info(f"Successfully processed: {bam_path.name}")
    
    # After all jobs are done, run the summary script
    logging.info("\n" + "="*80)
    logging.info("All samples processed. Generating summary report...")
    # Use your awk one-liner, but modified to parse JSON with a tool like `jq`
    summary_cmd = """
    jq -s '
      def avg(stream): stream as $s | reduce $s as $x (0; . + $x) / ($s|length);
      .[] | select(.mean != null) |
      {sample: .sample_name, mean: .mean, std_dev: .std_dev}
    ' """ + str(output_path) + """/*_stats.json | \
    (echo "Sample\tMean Frag Len\tStd Dev"; column -t -s $'\t')
    """
    # Note: This summary part requires `jq` to be installed. A pure awk version is also possible.
    # For simplicity, we can also just print a message telling the user where to find the files.
    logging.info(f"Individual JSON statistics files are located in: {output_path}")
    logging.info("="*80)


def main():
    import fileinput # Import here as it's part of the main logic now
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description='Calculates cDNA fragment length stats from BAM files by sampling intra-exonic reads.'
    )
    parser.add_argument('bam_list', nargs='?', default=None, 
                        help='A text file with BAM paths, one per line (reads from stdin if omitted).')
    parser.add_argument('-o', '--output-dir', required=True, 
                        help='Output directory for the generated JSON stats files.')
    parser.add_argument('-n', '--num-pairs', type=int, default=1000000,
                        help='Number of intra-exonic read pairs to sample (default: 1,000,000).')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Number of BAM files to process in parallel (default: 4).')
    parser.add_argument('--min_frag_len', type=int, default=50,
                        help='Minimum fragment length to consider (default: 50).')
    parser.add_argument('--max_frag_len', type=int, default=1000,
                        help='Maximum fragment length to consider (default: 1000).')
    parser.add_argument('--burn_in_fraction', type=float, default=0.1,
                        help='fraction of -n to skip at beginng (helps avoid positional effects on sorted BAMs)  (default: .1).')
    
    args = parser.parse_args()
    result = subprocess.run('which samtools', shell=True, capture_output=True, text=True, executable='/bin/bash')
    logging.info(f'{result.stdout}')
    process_bam_list(args.bam_list, args.output_dir, args.num_pairs, args.threads, args.min_frag_len, args.max_frag_len, args.burn_in_fraction)
    logging.info("Processing complete.")

if __name__ == '__main__':
    main()