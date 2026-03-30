#!/usr/bin/env python
"""
Cache management for the TE pipeline to enable parameter exploration.
"""

import os
import pickle
import hashlib
import logging
from datetime import datetime
import json


class PipelineCache:
    """Manages caching of featureCounts results for parameter exploration."""

    CACHE_VERSION = "1.0"

    def __init__(self, cache_dir="pipeline_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.load_metadata()

    def load_metadata(self):
        """Load or initialize cache metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": self.CACHE_VERSION,
                "created": datetime.now().isoformat(),
                "samples": {},
            }

    def save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_cache_key(self, bam_path, annotation_metadata, args):
        """Generate cache key including annotation checksums."""
        key_factors = {
            "bam_mtime": os.path.getmtime(bam_path),
            "bam_size": os.path.getsize(bam_path),
            # Use annotation checksums instead of GTF paths
            "annotation_gene_checksum": annotation_metadata["gene_checksum"],
            "annotation_te_checksum": annotation_metadata.get("te_checksum"),
            "annotation_creation": annotation_metadata["creation_date"],
            "annotation_n_transcripts": annotation_metadata["n_transcripts"],
            # Pipeline parameters
            "strandedness": args.strandedness,
            "paired_end": args.paired_end,
            "use_junctions": True,
        }

        # Create hash
        key_string = json.dumps(key_factors, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest(), key_factors

    def _file_checksum(self, filepath, chunk_size=65536):
        """Calculate MD5 checksum of a file."""
        if not filepath or not os.path.exists(filepath):
            return None

        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            # Only hash first and last chunks for large files
            f.seek(0)
            hasher.update(f.read(chunk_size))

            file_size = os.path.getsize(filepath)
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, os.SEEK_END)
                hasher.update(f.read(chunk_size))

        return hasher.hexdigest()

    def get_sample_cache_path(self, sample_name, cache_key):
        """Get the cache file path for a sample."""
        return os.path.join(self.cache_dir, f"{sample_name}_{cache_key}.pkl")

    def is_cache_valid(self, sample_name, bam_path, gtf_path, args):
        """Check if cache exists and is valid for given parameters."""
        cache_key, key_factors = self.get_cache_key(bam_path, gtf_path, args)
        cache_path = self.get_sample_cache_path(sample_name, cache_key)

        if not os.path.exists(cache_path):
            logging.info(f"[{sample_name}] No cache found")
            return False, cache_path, cache_key

        # Check if cache metadata matches
        if sample_name in self.metadata["samples"]:
            cached_factors = self.metadata["samples"][sample_name].get(
                "key_factors", {}
            )

            # Compare key factors
            for key, value in key_factors.items():
                if cached_factors.get(key) != value:
                    logging.info(
                        f"[{sample_name}] Cache invalidated due to {key} change"
                    )
                    return False, cache_path, cache_key

        logging.info(f"[{sample_name}] Valid cache found")
        return True, cache_path, cache_key

    def save_sample_cache(self, sample_name, cache_key, cache_data, key_factors):
        """Save cache data for a sample."""
        cache_path = self.get_sample_cache_path(sample_name, cache_key)

        # Add metadata to cache
        cache_data["_metadata"] = {
            "sample_name": sample_name,
            "cache_key": cache_key,
            "created": datetime.now().isoformat(),
            "pipeline_version": self.CACHE_VERSION,
        }

        # Save cache file
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata
        self.metadata["samples"][sample_name] = {
            "cache_key": cache_key,
            "cache_path": cache_path,
            "created": datetime.now().isoformat(),
            "key_factors": key_factors,
        }
        self.save_metadata()

        logging.info(f"[{sample_name}] Cache saved to {cache_path}")

    def load_sample_cache(self, cache_path):
        """Load cache data for a sample."""
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def update_sample_cache_with_multimapper_data(self, sample_name, multimapper_data):
        """Update existing cache with multimapper results."""
        # Try to find the cache file directly
        import glob

        pattern = os.path.join(self.cache_dir, f"{sample_name}_*.pkl")
        cache_files = glob.glob(pattern)

        if not cache_files:
            logging.warning(f"[{sample_name}] No cache files found")
            return False

        # Use the most recent cache file
        cache_path = max(cache_files, key=os.path.getmtime)

        # Load existing cache
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        # Update with multimapper data
        cache_data.update(multimapper_data)
        cache_data["has_multimapper_data"] = True
        cache_data["_metadata"]["multimapper_added"] = datetime.now().isoformat()

        # Save updated cache
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata if the sample exists
        self.load_metadata()  # Reload to be safe
        if sample_name in self.metadata["samples"]:
            self.metadata["samples"][sample_name]["has_multimapper_data"] = True
            self.save_metadata()

        logging.info(f"[{sample_name}] Updated cache with multimapper data")
        return True

    def clear_cache(self, sample_name=None):
        """Clear cache for a specific sample or all samples."""
        if sample_name:
            if sample_name in self.metadata["samples"]:
                cache_info = self.metadata["samples"][sample_name]
                if os.path.exists(cache_info["cache_path"]):
                    os.remove(cache_info["cache_path"])
                del self.metadata["samples"][sample_name]
                self.save_metadata()
                logging.info(f"Cleared cache for {sample_name}")
        else:
            # Clear all caches
            for sample in list(self.metadata["samples"].keys()):
                self.clear_cache(sample)
            logging.info("Cleared all caches")
