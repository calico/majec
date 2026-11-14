//! majec-core: High-performance EqC builder for MAJEC RNA-seq quantification.
//!
//! Replaces the pysam-based eqc_builder.py with a single-pass Rust BAM scanner
//! that builds junction-aware equivalence classes. Returns results as Python
//! dicts/lists for direct consumption by the MAJEC pipeline.

mod exon_index;
mod bam_processing;
mod junction_metrics;

use pyo3::prelude::*;

/// Python module entry point.
#[pymodule]
fn majec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<exon_index::ExonIntervalIndex>()?;
    m.add_function(wrap_pyfunction!(bam_processing::process_bam, m)?)?;
    m.add_function(wrap_pyfunction!(junction_metrics::build_junction_metrics, m)?)?;
    Ok(())
}
