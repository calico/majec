import argparse
import subprocess
import pandas as pd
import numpy as np
import os
import sys
import json
import re
from multiprocessing import Pool
import tempfile
import shutil

# --- (get_sample_name_from_bam function same as main pipeline) ---
def get_sample_name_from_bam(bam_path):
    """A robust function to derive a clean sample name from a BAM file path."""
    if not bam_path:
        return ""
    filename = os.path.basename(bam_path)
    suffix_tokens = [
        'Aligned', 'sortedByCoord', 'toTranscriptome', 'out', 'sortedByName', 
        'sorted', 'dedup', 'mkdup', 'pe', 'se', 'bam', 'sam', 'cram'
    ]
    pattern = r'(?:[._](?:' + '|'.join(suffix_tokens) + '))+$'
    sample_name = re.sub(pattern, '', filename, flags=re.IGNORECASE)
    common_prefixes = ['sorted_', 'dedup_']
    for prefix in common_prefixes:
        if sample_name.lower().startswith(prefix):
            sample_name = sample_name[len(prefix):]
            break
    return sample_name

def run_featurecounts_for_sample(args_tuple):
    """
    Worker function to run featureCounts on a single BAM and return its count Series.
    """
    bam_path, gtf_path, args = args_tuple
    sample_name = get_sample_name_from_bam(bam_path)
    print(f"  Processing: {sample_name}...")
    
    # Create a temporary directory for this specific job's output
    temp_dir = tempfile.mkdtemp(dir=args.tempdir)
    output_path = os.path.join(temp_dir, "fc_output.tsv")

    try:
        paired_end_flag = "-p" if args.paired_end else ""
        
        cmd = (f"featureCounts -a \"{gtf_path}\" -o \"{output_path}\" -s {args.strandedness} "
               f"-T 1 -t exon -g transcript_id {paired_end_flag} -Q 30 -O --fraction\"{bam_path}\"")

        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        # Load the simple count table
        counts_df = pd.read_csv(output_path, sep='\t', comment='#', usecols=['Geneid', bam_path])
        
        # Format into a pandas Series
        counts_series = counts_df.set_index('Geneid')[bam_path]
        counts_series.name = sample_name
        
        return counts_series

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: featureCounts failed for sample {sample_name}", file=sys.stderr)
        print(f"  BAM: {bam_path}", file=sys.stderr)
        print(f"  Stderr:\n{e.stderr}\n", file=sys.stderr)
        return None
    finally:
        # Ensure the temporary directory is always cleaned up
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate global expression binning thresholds for MAJEC pipeline. "
                    "This script runs featureCounts in parallel on a list of BAM files "
                    "in a memory-efficient way to generate a thresholds JSON file."
    )
    # --- ARGUMENT CHANGE: Mirroring the main pipeline ---
    parser.add_argument('--bams', nargs='+', required=True, help="List of coordinate-sorted input BAM files.")
    parser.add_argument('--gene_gtf', required=True, help="Path to the primary gene annotation GTF file.")
    parser.add_argument('--te_gtf', default=None, help="Path to the TE annotation GTF file (optional). If provided, it will be combined with the gene GTF.")
    # --- END CHANGE ---
    parser.add_argument('--output', required=True, help="Path for the output thresholds JSON file (e.g., 'global_bins.json').")
    parser.add_argument('--threads', type=int, default=4, help="Number of BAM files to process in parallel (default: 4).")
    parser.add_argument('--strandedness', default=2, type=int, choices=[0, 1, 2], help="Strand specificity (default: 2).")
    parser.add_argument('--paired_end', action='store_true', help="Data is paired-end.")
    parser.add_argument('--tempdir', help="Directory for temporary files.")
    
    args = parser.parse_args()

    print(f"--- Starting Global Bin Threshold Calculation ---")

    # Use a top-level temporary directory for the combined GTF if needed
    with tempfile.TemporaryDirectory(dir=args.tempdir) as temp_dir_main:
        
        # --- NEW LOGIC: Handle single or combined GTF ---
        if args.te_gtf:
            print("Combining gene and TE GTF files into a temporary file...")
            gtf_to_use = os.path.join(temp_dir_main, "combined_for_bins.gtf")
            try:
                # Use subprocess for robust handling of large files
                with open(gtf_to_use, 'wb') as outfile:
                    subprocess.run(['cat', args.gene_gtf, args.te_gtf], stdout=outfile, check=True)
                print(f"  Combined GTF created at: {gtf_to_use}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"FATAL: Failed to combine GTF files. Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # If no TE GTF, just use the gene GTF directly
            gtf_to_use = args.gene_gtf
        # --- END NEW LOGIC ---

        print(f"Processing {len(args.bams)} BAM files using {args.threads} parallel jobs.")

        # Prepare arguments for each worker, using our single GTF path
        pool_args = [(os.path.abspath(bam), gtf_to_use, args) for bam in args.bams]

        # ... (The rest of the script: Pool, results processing, threshold calculation) ...
        # ... is exactly the same and does not need to be changed. ...
        # It will correctly use the 'gtf_to_use' variable.
        with Pool(processes=args.threads) as pool:
            results = pool.map(run_featurecounts_for_sample, pool_args)

        successful_results = [res for res in results if res is not None]
        if len(successful_results) != len(args.bams):
            print(f"\nWARNING: {len(args.bams) - len(successful_results)} sample(s) failed featureCounts. "
                  "Thresholds will be calculated on the remaining samples.")
            if not successful_results:
                print("FATAL: All samples failed. Exiting.", file=sys.stderr)
                sys.exit(1)

        print("\nCombining count data into a single matrix...")
        global_counts_df = pd.concat(successful_results, axis=1)
        global_counts_df.fillna(0, inplace=True)
        
        print("Calculating expression percentiles...")
        all_counts = global_counts_df.values.flatten()
        expressed_counts = all_counts[all_counts > 0]

        if len(expressed_counts) == 0:
            print("FATAL: No expressed features found across any samples. Cannot calculate thresholds.", file=sys.stderr)
            sys.exit(1)
            
        low_threshold = np.percentile(expressed_counts, 25)
        high_threshold = np.percentile(expressed_counts, 75)

        print(f"  Low threshold (25th percentile): {low_threshold:.2f}")
        print(f"  High threshold (75th percentile): {high_threshold:.2f}")

        output_data = {
            'global_thresholds': {
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'samples_processed': [s.name for s in successful_results]
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n--- Success! ---")
        print(f"Global binning information saved to: {args.output}")
        print(f"You can now use this file with the --use_bins flag in the main MAJEC pipeline.")

if __name__ == '__main__':
    main()