use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use ago_control_plane::envelope::{
    AgoCAction, AgoCBudget, AgoCCompletion, AgoCEvent, ControlPlane, COMPLETION_OK,
    EVIDENCE_FRAME_OBSERVATION, OP_REFINE, SERVICE_P2_REFINEMENT,
};

struct CountingAllocator;

static TRACK_ALLOCS: AtomicBool = AtomicBool::new(false);
static HOT_PATH_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static HOT_PATH_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCS.load(Ordering::Relaxed) {
            HOT_PATH_ALLOCS.fetch_add(1, Ordering::Relaxed);
            HOT_PATH_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn read_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.split_whitespace().next()?;
            return kb_str.parse::<u64>().ok();
        }
    }
    None
}

fn main() {
    let mut cp = ControlPlane::new(4, 8);
    cp.set_budget(AgoCBudget {
        now_ms: 0.0,
        next_deadline_ms: 16.5,
        reserved_ms: 0.0,
        inflight_ms: 0.0,
        safety_margin_ms: 0.5,
    });

    let mut out = [AgoCAction::default(); 4];
    let mut comp_buf = [AgoCCompletion::default(); 4];

    // Warmup
    for i in 0u64..200 {
        let ev = AgoCEvent {
            kind: EVIDENCE_FRAME_OBSERVATION,
            region_id: (i % 64) + 1,
            magnitude: 0.5,
            timestamp_ms: i as f64,
            ..Default::default()
        };
        let act = AgoCAction {
            action_id: i + 1,
            region_id: (i % 64) + 1,
            service_class: SERVICE_P2_REFINEMENT,
            op: OP_REFINE,
            requires_gpu: 1,
            expected_gain: ((i % 11) + 1) as f32,
            estimated_cost_ms: 0.5,
            ..Default::default()
        };
        cp.ingest_events(&[ev]);
        cp.enqueue_actions(&[act]);
        let n = cp.schedule_batch(&mut out) as usize;
        for j in 0..n {
            comp_buf[j] = AgoCCompletion {
                action_id: out[j].action_id,
                region_id: out[j].region_id,
                op: out[j].op,
                service_class: out[j].service_class,
                status: COMPLETION_OK,
                actual_cost_ms: 0.5,
                completed_at_ms: (i as f64) + 0.5,
                ..Default::default()
            };
        }
        cp.ingest_completions(&comp_buf[..n]);
    }

    let n_ticks = 2000usize;
    let mut lat_us = Vec::with_capacity(n_ticks);
    let rss_before_kb = read_rss_kb().unwrap_or(0);
    HOT_PATH_ALLOCS.store(0, Ordering::SeqCst);
    HOT_PATH_BYTES.store(0, Ordering::SeqCst);
    for i in 0u64..(n_ticks as u64) {
        let ev = AgoCEvent {
            kind: EVIDENCE_FRAME_OBSERVATION,
            region_id: (i % 64) + 1,
            magnitude: 0.5,
            timestamp_ms: (1000 + i) as f64,
            ..Default::default()
        };
        let act = AgoCAction {
            action_id: 1000 + i,
            region_id: (i % 64) + 1,
            service_class: SERVICE_P2_REFINEMENT,
            op: OP_REFINE,
            requires_gpu: 1,
            expected_gain: ((i % 11) + 1) as f32,
            estimated_cost_ms: 0.5,
            ..Default::default()
        };
        let t0 = Instant::now();
        TRACK_ALLOCS.store(true, Ordering::SeqCst);
        cp.ingest_events(&[ev]);
        cp.enqueue_actions(&[act]);
        let n = cp.schedule_batch(&mut out) as usize;
        for j in 0..n {
            comp_buf[j] = AgoCCompletion {
                action_id: out[j].action_id,
                region_id: out[j].region_id,
                op: out[j].op,
                service_class: out[j].service_class,
                status: COMPLETION_OK,
                actual_cost_ms: 0.5,
                completed_at_ms: ((1000 + i) as f64) + 0.5,
                ..Default::default()
            };
        }
        cp.ingest_completions(&comp_buf[..n]);
        TRACK_ALLOCS.store(false, Ordering::SeqCst);
        let dt_us = t0.elapsed().as_nanos() as f64 * 1e-3;
        lat_us.push(dt_us);
    }

    let hot_allocs = HOT_PATH_ALLOCS.load(Ordering::SeqCst);
    let hot_bytes = HOT_PATH_BYTES.load(Ordering::SeqCst);
    let rss_after_kb = read_rss_kb().unwrap_or(0);
    assert_eq!(
        hot_allocs, 0,
        "Steady-state AGO control-plane hot path must perform 0 heap allocations (got {} allocs, {} bytes)",
        hot_allocs, hot_bytes
    );

    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lat_us[(n_ticks * 50) / 100];
    let p95 = lat_us[(n_ticks * 95) / 100];
    let p99 = lat_us[(n_ticks * 99) / 100];
    let sum = cp.resource_summary();
    println!(
        "[Rust ControlPlane bench_replay N={}] p50={:.3}us p95={:.3}us p99={:.3}us admitted={} completed={} mean_e2m={:.3}ms hot_allocs={} hot_bytes={} rss_before_kb={} rss_after_kb={} rss_delta_kb={}",
        n_ticks,
        p50,
        p95,
        p99,
        sum.total_admitted,
        sum.total_completed,
        sum.mean_event_to_map_correction_ms,
        hot_allocs,
        hot_bytes,
        rss_before_kb,
        rss_after_kb,
        rss_after_kb.saturating_sub(rss_before_kb),
    );
}
