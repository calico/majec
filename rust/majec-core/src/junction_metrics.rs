//! Build pipeline-compatible junction_metrics from aggregated junction counts.
//!
//! This replicates the Python `build_junction_metrics_from_counts()` logic but
//! runs in Rust for consistency (called once per sample, not per-read — so the
//! speedup here is minor, but it keeps all BAM-derived computation in one place).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Per-transcript junction metrics, matching the Python dict structure.
struct TranscriptJunctionMetrics {
    junction_read_count: u64,
    unique_junction_reads: u64,
    shared_junction_reads: u64,
    observed_junctions: HashMap<u32, u64>, // junction_int -> count
    n_unique_junctions: u32,
    n_shared_junctions: u32,
}

impl TranscriptJunctionMetrics {
    fn new() -> Self {
        TranscriptJunctionMetrics {
            junction_read_count: 0,
            unique_junction_reads: 0,
            shared_junction_reads: 0,
            observed_junctions: HashMap::new(),
            n_unique_junctions: 0,
            n_shared_junctions: 0,
        }
    }

    fn to_pydict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("junction_read_count", self.junction_read_count).unwrap();
        d.set_item("unique_junction_reads", self.unique_junction_reads).unwrap();
        d.set_item("shared_junction_reads", self.shared_junction_reads).unwrap();

        let obs = PyDict::new(py);
        for (&junc_int, &count) in &self.observed_junctions {
            obs.set_item(junc_int, count).unwrap();
        }
        d.set_item("observed_junctions", obs).unwrap();

        d.set_item("n_unique_junctions", self.n_unique_junctions).unwrap();
        d.set_item("n_shared_junctions", self.n_shared_junctions).unwrap();
        d
    }
}

/// Build junction_metrics from aggregated junction counts.
///
/// Args:
///     junction_counts: {unstranded_key: count} from process_bam
///     junction_map: {junction_int: {"transcripts": [int], "n_transcripts": int}}
///     junction_int_to_string: {junction_int: stranded_key_string}
///     junction_string_to_int: {stranded_key_string: junction_int}
///
/// Returns:
///     {transcript_int: {junction metrics dict}}
#[pyfunction]
pub fn build_junction_metrics(
    py: Python<'_>,
    junction_counts: HashMap<String, u64>,
    junction_map: HashMap<u32, HashMap<String, Py<PyAny>>>,
    _junction_int_to_string: HashMap<u32, String>,
    junction_string_to_int: HashMap<String, u32>,
) -> PyResult<Py<PyAny>> {
    let mut metrics: HashMap<u32, TranscriptJunctionMetrics> = HashMap::new();

    for (unstranded_key, count) in &junction_counts {
        if *count == 0 {
            continue;
        }

        // Try both stranded versions
        for strand in &["+", "-"] {
            let stranded_key = format!("{}:{}", unstranded_key, strand);

            if let Some(&junction_id) = junction_string_to_int.get(&stranded_key) {
                if let Some(info) = junction_map.get(&junction_id) {
                    // Extract transcripts list and n_transcripts from the Python dict
                    let transcripts: Vec<u32> = Python::attach(|py| {
                        if let Some(t) = info.get("transcripts") {
                            t.extract::<Vec<u32>>(py).unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    });

                    let n_transcripts: u32 = Python::attach(|py| {
                        if let Some(n) = info.get("n_transcripts") {
                            n.extract::<u32>(py).unwrap_or(1)
                        } else {
                            1
                        }
                    });

                    for &transcript in &transcripts {
                        let m = metrics.entry(transcript).or_insert_with(TranscriptJunctionMetrics::new);
                        m.junction_read_count += count;

                        if n_transcripts == 1 {
                            m.unique_junction_reads += count;
                        } else {
                            m.shared_junction_reads += count;
                        }

                        if !m.observed_junctions.contains_key(&junction_id) {
                            m.observed_junctions.insert(junction_id, 0);
                            if n_transcripts == 1 {
                                m.n_unique_junctions += 1;
                            } else {
                                m.n_shared_junctions += 1;
                            }
                        }
                        *m.observed_junctions.get_mut(&junction_id).unwrap() += count;
                    }

                    break; // Don't double-count on both strands
                }
            }
        }
    }

    // Convert to Python dict
    let result = PyDict::new(py);
    for (transcript_int, m) in &metrics {
        result.set_item(*transcript_int, m.to_pydict(py))?;
    }

    Ok(result.into())
}
