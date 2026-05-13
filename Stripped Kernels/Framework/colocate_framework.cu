/**
 * Kernel Colocation Framework
 *
 * Goal: Profile a set of CUDA kernels, collect resource usage metrics,
 * and decide which pairs (or groups) are good candidates to run concurrently
 * on the same GPU via CUDA streams.
 *
 * Pipeline:
 *   1. Register kernels + metadata
 *   2. Profile each kernel in isolation
 *   3. Score every pair for colocation compatibility
 *   4. Select best pairs / schedule
 *   5. Run colocated and validate result
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────────────────────
// 1.  CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────────

#define MAX_KERNELS        16
#define MAX_NAME_LEN       64
#define PROFILE_RUNS       10       // warm-up + measurement iterations
#define SCORE_THRESHOLD    0.6f     // minimum compatibility score to colocate


// ─────────────────────────────────────────────────────────────────────────────
// 2.  DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resource profile collected for a single kernel when run in isolation.
 * Extend with nvml / cupti counters as needed.
 */
typedef struct {
    float    avg_runtime_ms;        // average wall time
    float    sm_occupancy;          // fraction of SMs actively used  [0,1]
    float    mem_bandwidth_util;    // fraction of peak memory BW used [0,1]
    float    compute_util;          // fraction of peak FLOP/s used    [0,1]
    size_t   shared_mem_bytes;      // shared memory per block
    int      registers_per_thread;  // register file pressure
    int      blocks_per_sm;         // achieved blocks per SM
} KernelProfile;

/**
 * Descriptor for one kernel registered with the framework.
 */
typedef struct {
    char     name[MAX_NAME_LEN];

    // Launch configuration
    void     (*kernel_fn)(void);    // type-erased; cast before calling
    void    **args;                 // kernel arguments array (cudaLaunchKernel style)
    dim3     grid;
    dim3     block;
    size_t   shared_mem;

    // Filled in by the profiler
    KernelProfile profile;
    int           profiled;         // bool: has been profiled?
} KernelDescriptor;

/**
 * Score and metadata for a candidate colocation pair.
 */
typedef struct {
    int   idx_a;
    int   idx_b;
    float score;                    // higher = better colocation candidate
    float expected_speedup;         // estimated vs sequential execution
} ColocPair;

/**
 * Top-level framework state.
 */
typedef struct {
    KernelDescriptor kernels[MAX_KERNELS];
    int              num_kernels;

    ColocPair        pairs[MAX_KERNELS * MAX_KERNELS];
    int              num_pairs;

    cudaStream_t     streams[MAX_KERNELS]; // one stream per registered kernel
} ColocFramework;


// ─────────────────────────────────────────────────────────────────────────────
// 3.  FRAMEWORK LIFECYCLE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the framework.  Call once before anything else.
 */
void framework_init(ColocFramework *fw)
{
    memset(fw, 0, sizeof(*fw));
    printf("[framework] initialized\n");
}

/**
 * Tear down streams and free any framework-owned resources.
 */
void framework_destroy(ColocFramework *fw)
{
    for (int i = 0; i < fw->num_kernels; i++) {
        if (fw->streams[i]) {
            cudaStreamDestroy(fw->streams[i]);
            fw->streams[i] = NULL;
        }
    }
    printf("[framework] destroyed\n");
}


// ─────────────────────────────────────────────────────────────────────────────
// 4.  KERNEL REGISTRATION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Register a kernel with the framework.
 *
 * @param fw         Framework instance
 * @param name       Human-readable label
 * @param kernel_fn  Pointer to the __global__ function (cast to void(*)(void))
 * @param args       Argument array for cudaLaunchKernel
 * @param grid       Grid dimensions
 * @param block      Block dimensions
 * @param shared_mem Shared memory per block in bytes
 * @return           Index of the registered kernel, or -1 on failure
 */
int framework_register_kernel(ColocFramework *fw,
                              const char     *name,
                              void          (*kernel_fn)(void),
                              void          **args,
                              dim3           grid,
                              dim3           block,
                              size_t         shared_mem)
{
    if (fw->num_kernels >= MAX_KERNELS) {
        fprintf(stderr, "[framework] ERROR: MAX_KERNELS reached\n");
        return -1;
    }

    int idx = fw->num_kernels++;
    KernelDescriptor *kd = &fw->kernels[idx];

    strncpy(kd->name, name, MAX_NAME_LEN - 1);
    kd->kernel_fn  = kernel_fn;
    kd->args       = args;
    kd->grid       = grid;
    kd->block      = block;
    kd->shared_mem = shared_mem;
    kd->profiled   = 0;

    cudaStreamCreate(&fw->streams[idx]);

    printf("[framework] registered kernel[%d]: %s\n", idx, name);
    return idx;
}


// ─────────────────────────────────────────────────────────────────────────────
// 7.  COLOCATED EXECUTION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Launch kernels a and b concurrently on separate streams.
 * Blocks until both finish.
 *
 * TODO: extend to launch full groups, not just pairs.
 */
void framework_run_colocated(ColocFramework *fw, int idx_a, int idx_b)
{
    KernelDescriptor *ka = &fw->kernels[idx_a];
    KernelDescriptor *kb = &fw->kernels[idx_b];

    printf("[exec] colocating '%s' and '%s'\n", ka->name, kb->name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, fw->streams[idx_a]);

    // Launch both without synchronizing between them
    cudaLaunchKernel((const void *)ka->kernel_fn,
                     ka->grid, ka->block, ka->args,
                     ka->shared_mem, fw->streams[idx_a]);

    cudaLaunchKernel((const void *)kb->kernel_fn,
                     kb->grid, kb->block, kb->args,
                     kb->shared_mem, fw->streams[idx_b]);

    // Wait for both streams
    cudaStreamSynchronize(fw->streams[idx_a]);
    cudaStreamSynchronize(fw->streams[idx_b]);

    cudaEventRecord(stop, fw->streams[idx_a]);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[exec] colocated runtime: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// ─────────────────────────────────────────────────────────────────────────────
// 8.  REPORTING
// ─────────────────────────────────────────────────────────────────────────────

void framework_print_report(ColocFramework *fw)
{
    printf("\n════════════════════════════════════════\n");
    printf(" Kernel Colocation Report\n");
    printf("════════════════════════════════════════\n");

    printf("\nProfiling results:\n");
    for (int i = 0; i < fw->num_kernels; i++) {
        KernelDescriptor *kd = &fw->kernels[i];
        if (!kd->profiled) continue;
        printf("  [%d] %-20s  avg=%.3f ms  SM_occ=%.2f  mem_bw=%.2f  compute=%.2f\n",
               i, kd->name,
               kd->profile.avg_runtime_ms,
               kd->profile.sm_occupancy,
               kd->profile.mem_bandwidth_util,
               kd->profile.compute_util);
    }

    printf("\nTop colocation candidates:\n");
    int shown = 0;
    for (int i = 0; i < fw->num_pairs && shown < 5; i++, shown++) {
        ColocPair *cp = &fw->pairs[i];
        printf("  #%d  (%s + %s)  score=%.3f  speedup=%.2fx  %s\n",
               shown + 1,
               fw->kernels[cp->idx_a].name,
               fw->kernels[cp->idx_b].name,
               cp->score,
               cp->expected_speedup,
               cp->score >= SCORE_THRESHOLD ? "✓ RECOMMENDED" : "✗ below threshold");
    }
    printf("════════════════════════════════════════\n\n");
}


// ─────────────────────────────────────────────────────────────────────────────
// 9.  EXAMPLE USAGE  (remove / adapt for your own kernels)
// ─────────────────────────────────────────────────────────────────────────────

// --- Example kernel A: compute-bound ---
__global__ void kernel_compute(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        // Fake compute-heavy work
        for (int k = 0; k < 100; k++) v = sinf(v) + cosf(v);
        data[i] = v;
    }
}

// --- Example kernel B: memory-bound ---
__global__ void kernel_memcopy(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2.0f;
}

int main(void)
{
    const int N = 1 << 20;

    float *d_data, *d_src, *d_dst;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_src,  N * sizeof(float));
    cudaMalloc(&d_dst,  N * sizeof(float));

    // ── Build argument arrays for cudaLaunchKernel ───────────────────────────
    int n = N;
    void *args_compute[] = { &d_data, &n };
    void *args_memcopy[] = { &d_src, &d_dst, &n };

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    // ── Framework setup ──────────────────────────────────────────────────────
    ColocFramework fw;
    framework_init(&fw);

    framework_register_kernel(&fw, "kernel_compute",
                              (void(*)(void))kernel_compute,
                              args_compute, grid, block, 0);

    framework_register_kernel(&fw, "kernel_memcopy",
                              (void(*)(void))kernel_memcopy,
                              args_memcopy, grid, block, 0);

    // ── Profile → Score → Rank → Execute ────────────────────────────────────
    framework_profile_all(&fw);
    framework_score_pairs(&fw);
    framework_rank_pairs(&fw);
    framework_print_report(&fw);

    ColocPair *best = framework_best_pair(&fw);
    if (best) {
        framework_run_colocated(&fw, best->idx_a, best->idx_b);
    } else {
        printf("[main] No pair meets the colocation threshold; running sequentially.\n");
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    framework_destroy(&fw);
    cudaFree(d_data);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
