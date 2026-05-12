#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void *CUstream;
typedef unsigned long long cuuint64_t;
typedef int CUdriverProcAddressQueryResult;

typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

typedef struct {
    uint64_t calls;
    uint64_t bytes;
    uint64_t ns;
} counter_t;

static counter_t g_cu_mem_alloc_async;
static counter_t g_cu_mem_free_async;
static counter_t g_cu_mem_alloc_v2;
static counter_t g_cu_mem_free_v2;
static counter_t g_cu_memset_d8_async;
static counter_t g_cuda_malloc;
static counter_t g_cuda_free;
static counter_t g_cuda_malloc_async;
static counter_t g_cuda_free_async;
static counter_t g_cuda_memset_async;
static counter_t g_cuda_launch_kernel;
static counter_t g_cu_launch_kernel;
static counter_t g_cu_launch_kernel_ex;
static counter_t g_cu_memcpy_htod_async;
static counter_t g_cu_memcpy_dtoh_async;
static counter_t g_cu_get_proc_address;
static counter_t g_cu_get_proc_address_replaced;

static void *g_cu_mem_alloc_async_from_getproc;
static void *g_cu_mem_free_async_from_getproc;
static void *g_cu_memset_d8_async_from_getproc;
static void *g_cu_mem_alloc_async_ptsz_from_getproc;
static void *g_cu_mem_free_async_ptsz_from_getproc;
static void *g_cu_memset_d8_async_ptsz_from_getproc;

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t n, CUstream hStream);
CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream);
CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t n, CUstream hStream);

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void record(counter_t *counter, uint64_t bytes, uint64_t start_ns) {
    __sync_fetch_and_add(&counter->calls, 1);
    __sync_fetch_and_add(&counter->bytes, bytes);
    __sync_fetch_and_add(&counter->ns, now_ns() - start_ns);
}

static void *must_next(const char *name) {
    void *fn = dlsym(RTLD_NEXT, name);
    if (!fn) {
        fprintf(stderr, "[cuda-api-counter] failed to resolve %s: %s\n", name, dlerror());
        abort();
    }
    return fn;
}

static void maybe_replace_driver_proc(const char *symbol, void **pfn) {
    if (!symbol || !pfn || !*pfn) return;
    if (strcmp(symbol, "cuMemAllocAsync") == 0) {
        g_cu_mem_alloc_async_from_getproc = *pfn;
        *pfn = (void *)&cuMemAllocAsync;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    } else if (strcmp(symbol, "cuMemAllocAsync_ptsz") == 0) {
        g_cu_mem_alloc_async_ptsz_from_getproc = *pfn;
        *pfn = (void *)&cuMemAllocAsync_ptsz;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    } else if (strcmp(symbol, "cuMemFreeAsync") == 0) {
        g_cu_mem_free_async_from_getproc = *pfn;
        *pfn = (void *)&cuMemFreeAsync;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    } else if (strcmp(symbol, "cuMemFreeAsync_ptsz") == 0) {
        g_cu_mem_free_async_ptsz_from_getproc = *pfn;
        *pfn = (void *)&cuMemFreeAsync_ptsz;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    } else if (strcmp(symbol, "cuMemsetD8Async") == 0) {
        g_cu_memset_d8_async_from_getproc = *pfn;
        *pfn = (void *)&cuMemsetD8Async;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    } else if (strcmp(symbol, "cuMemsetD8Async_ptsz") == 0) {
        g_cu_memset_d8_async_ptsz_from_getproc = *pfn;
        *pfn = (void *)&cuMemsetD8Async_ptsz;
        record(&g_cu_get_proc_address_replaced, 0, now_ns());
    }
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr *, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_mem_alloc_async_from_getproc
                      ? (fn_t)g_cu_mem_alloc_async_from_getproc
                      : (fn_t)must_next("cuMemAllocAsync");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr, bytesize, hStream);
    record(&g_cu_mem_alloc_async, bytesize, start);
    return result;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_mem_free_async_from_getproc
                      ? (fn_t)g_cu_mem_free_async_from_getproc
                      : (fn_t)must_next("cuMemFreeAsync");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr, hStream);
    record(&g_cu_mem_free_async, 0, start);
    return result;
}

CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr *, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_mem_alloc_async_ptsz_from_getproc
                      ? (fn_t)g_cu_mem_alloc_async_ptsz_from_getproc
                      : (fn_t)must_next("cuMemAllocAsync_ptsz");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr, bytesize, hStream);
    record(&g_cu_mem_alloc_async, bytesize, start);
    return result;
}

CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_mem_free_async_ptsz_from_getproc
                      ? (fn_t)g_cu_mem_free_async_ptsz_from_getproc
                      : (fn_t)must_next("cuMemFreeAsync_ptsz");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr, hStream);
    record(&g_cu_mem_free_async, 0, start);
    return result;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    typedef CUresult (*fn_t)(CUdeviceptr *, size_t);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuMemAlloc_v2");
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr, bytesize);
    record(&g_cu_mem_alloc_v2, bytesize, start);
    return result;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    typedef CUresult (*fn_t)(CUdeviceptr);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuMemFree_v2");
    uint64_t start = now_ns();
    CUresult result = real_fn(dptr);
    record(&g_cu_mem_free_v2, 0, start);
    return result;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t n, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr, unsigned char, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_memset_d8_async_from_getproc
                      ? (fn_t)g_cu_memset_d8_async_from_getproc
                      : (fn_t)must_next("cuMemsetD8Async");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dstDevice, uc, n, hStream);
    record(&g_cu_memset_d8_async, n, start);
    return result;
}

CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t n, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr, unsigned char, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) {
        real_fn = g_cu_memset_d8_async_ptsz_from_getproc
                      ? (fn_t)g_cu_memset_d8_async_ptsz_from_getproc
                      : (fn_t)must_next("cuMemsetD8Async_ptsz");
    }
    uint64_t start = now_ns();
    CUresult result = real_fn(dstDevice, uc, n, hStream);
    record(&g_cu_memset_d8_async, n, start);
    return result;
}

int cudaMallocAsync(void **devPtr, size_t size, void *stream) {
    typedef int (*fn_t)(void **, size_t, void *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaMallocAsync");
    uint64_t start = now_ns();
    int result = real_fn(devPtr, size, stream);
    record(&g_cuda_malloc_async, size, start);
    return result;
}

int cudaFreeAsync(void *devPtr, void *stream) {
    typedef int (*fn_t)(void *, void *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaFreeAsync");
    uint64_t start = now_ns();
    int result = real_fn(devPtr, stream);
    record(&g_cuda_free_async, 0, start);
    return result;
}

int cudaMemsetAsync(void *devPtr, int value, size_t count, void *stream) {
    typedef int (*fn_t)(void *, int, size_t, void *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaMemsetAsync");
    uint64_t start = now_ns();
    int result = real_fn(devPtr, value, count, stream);
    record(&g_cuda_memset_async, count, start);
    return result;
}

int cudaMalloc(void **devPtr, size_t size) {
    typedef int (*fn_t)(void **, size_t);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaMalloc");
    uint64_t start = now_ns();
    int result = real_fn(devPtr, size);
    record(&g_cuda_malloc, size, start);
    return result;
}

int cudaFree(void *devPtr) {
    typedef int (*fn_t)(void *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaFree");
    uint64_t start = now_ns();
    int result = real_fn(devPtr);
    record(&g_cuda_free, 0, start);
    return result;
}

int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                     size_t sharedMem, void *stream) {
    typedef int (*fn_t)(const void *, dim3, dim3, void **, size_t, void *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cudaLaunchKernel");
    uint64_t start = now_ns();
    int result = real_fn(func, gridDim, blockDim, args, sharedMem, stream);
    record(&g_cuda_launch_kernel, 0, start);
    return result;
}

CUresult cuLaunchKernel(void *f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX,
                        unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
    typedef CUresult (*fn_t)(void *, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int,
                             unsigned int, CUstream, void **, void **);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuLaunchKernel");
    uint64_t start = now_ns();
    CUresult result = real_fn(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                              blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    record(&g_cu_launch_kernel, 0, start);
    return result;
}

CUresult cuLaunchKernelEx(const void *config, void *f, void **kernelParams, void **extra) {
    typedef CUresult (*fn_t)(const void *, void *, void **, void **);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuLaunchKernelEx");
    uint64_t start = now_ns();
    CUresult result = real_fn(config, f, kernelParams, extra);
    record(&g_cu_launch_kernel_ex, 0, start);
    return result;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t byteCount, CUstream hStream) {
    typedef CUresult (*fn_t)(CUdeviceptr, const void *, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuMemcpyHtoDAsync_v2");
    uint64_t start = now_ns();
    CUresult result = real_fn(dstDevice, srcHost, byteCount, hStream);
    record(&g_cu_memcpy_htod_async, byteCount, start);
    return result;
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t byteCount, CUstream hStream) {
    typedef CUresult (*fn_t)(void *, CUdeviceptr, size_t, CUstream);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuMemcpyDtoHAsync_v2");
    uint64_t start = now_ns();
    CUresult result = real_fn(dstHost, srcDevice, byteCount, hStream);
    record(&g_cu_memcpy_dtoh_async, byteCount, start);
    return result;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
    typedef CUresult (*fn_t)(const char *, void **, int, cuuint64_t,
                             CUdriverProcAddressQueryResult *);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuGetProcAddress_v2");
    uint64_t start = now_ns();
    CUresult result = real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
    record(&g_cu_get_proc_address, 0, start);
    if (result == 0) {
        maybe_replace_driver_proc(symbol, pfn);
    }
    return result;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
    typedef CUresult (*fn_t)(const char *, void **, int, cuuint64_t);
    static fn_t real_fn;
    if (!real_fn) real_fn = (fn_t)must_next("cuGetProcAddress");
    uint64_t start = now_ns();
    CUresult result = real_fn(symbol, pfn, cudaVersion, flags);
    record(&g_cu_get_proc_address, 0, start);
    if (result == 0) {
        maybe_replace_driver_proc(symbol, pfn);
    }
    return result;
}

static void print_counter(const char *name, const counter_t *counter) {
    fprintf(stderr,
            "[cuda-api-counter] %s calls=%llu bytes=%llu total_ns=%llu\n",
            name,
            (unsigned long long)counter->calls,
            (unsigned long long)counter->bytes,
            (unsigned long long)counter->ns);
}

__attribute__((destructor)) static void print_cuda_api_counters(void) {
    print_counter("cuMemAllocAsync", &g_cu_mem_alloc_async);
    print_counter("cuMemFreeAsync", &g_cu_mem_free_async);
    print_counter("cuMemAlloc_v2", &g_cu_mem_alloc_v2);
    print_counter("cuMemFree_v2", &g_cu_mem_free_v2);
    print_counter("cuMemsetD8Async", &g_cu_memset_d8_async);
    print_counter("cudaMalloc", &g_cuda_malloc);
    print_counter("cudaFree", &g_cuda_free);
    print_counter("cudaMallocAsync", &g_cuda_malloc_async);
    print_counter("cudaFreeAsync", &g_cuda_free_async);
    print_counter("cudaMemsetAsync", &g_cuda_memset_async);
    print_counter("cudaLaunchKernel", &g_cuda_launch_kernel);
    print_counter("cuLaunchKernel", &g_cu_launch_kernel);
    print_counter("cuLaunchKernelEx", &g_cu_launch_kernel_ex);
    print_counter("cuMemcpyHtoDAsync_v2", &g_cu_memcpy_htod_async);
    print_counter("cuMemcpyDtoHAsync_v2", &g_cu_memcpy_dtoh_async);
    print_counter("cuGetProcAddress", &g_cu_get_proc_address);
    print_counter("cuGetProcAddress_replaced", &g_cu_get_proc_address_replaced);
}
