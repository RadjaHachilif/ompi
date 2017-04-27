#include "coll_adapt.h"
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "opal/mca/common/cuda/common_cuda.h"
#include "opal/mca/installdirs/installdirs.h"
#include <dlfcn.h>
#include <unistd.h>

static coll_adapt_cuda_function_table_t coll_adapt_cuda_table;
static void *coll_adapt_cuda_kernel_handle = NULL;
static char *coll_adapt_cuda_kernel_lib = NULL;

#define COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN(handle, fname)            \
    do {                                                                            \
        char* _error;                                                               \
        *(void **)(&(coll_adapt_cuda_table.fname ## _p)) = dlsym((handle), # fname);    \
        if(NULL != (_error = dlerror()) )  {                                        \
            opal_output(0, "Finding %s error: %s\n", # fname, _error);              \
            coll_adapt_cuda_table.fname ## _p = NULL;                                   \
            return OMPI_ERROR;                                                      \
        }                                                                           \
    } while (0)

int coll_adapt_cuda_init(void)
{
    if (coll_adapt_cuda_kernel_handle == NULL) {
        if( NULL != coll_adapt_cuda_kernel_lib )
            free(coll_adapt_cuda_kernel_lib);
        asprintf(&coll_adapt_cuda_kernel_lib, "%s/%s", opal_install_dirs.libdir, "mca_coll_adapt_cuda_kernel.so");

        coll_adapt_cuda_kernel_handle = dlopen(coll_adapt_cuda_kernel_lib , RTLD_LAZY);
        if (!coll_adapt_cuda_kernel_handle) {
            opal_output( 0, "Failed to load %s library: error %s\n", coll_adapt_cuda_kernel_lib, dlerror());
            coll_adapt_cuda_kernel_handle = NULL;
            return OMPI_ERROR;
        }
    
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_init );
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_fini );
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_kernel_handle, coll_adapt_cuda_is_gpu_buffer );
    
        coll_adapt_cuda_table.coll_adapt_cuda_init_p();
        mca_coll_adapt_component.coll_adapt_cuda_enabled = 1;
        
        mca_coll_adapt_component_t *cs = &mca_coll_adapt_component;
        cs->pined_cpu_mpool = coll_adapt_cuda_mpool_create();
        
        opal_output( 0, "coll_adapt_cuda_init done\n");
    }
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_fini(void)
{
    if (coll_adapt_cuda_kernel_handle != NULL) {
        coll_adapt_cuda_table.coll_adapt_cuda_fini_p();
        
        coll_adapt_cuda_table.coll_adapt_cuda_init_p = NULL;
        coll_adapt_cuda_table.coll_adapt_cuda_fini_p = NULL;
        coll_adapt_cuda_table.coll_adapt_cuda_is_gpu_buffer_p = NULL;
        
        dlclose(coll_adapt_cuda_kernel_handle);
        coll_adapt_cuda_kernel_handle = NULL;
        if( NULL != coll_adapt_cuda_kernel_lib ) {
            free(coll_adapt_cuda_kernel_lib);
        }
        coll_adapt_cuda_kernel_lib = NULL;
        mca_coll_adapt_component.coll_adapt_cuda_enabled = 0;
        opal_output( 0, "coll_adapt_cuda_fini done\n");
    }
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_is_gpu_buffer(const void *ptr)
{
    return coll_adapt_cuda_table.coll_adapt_cuda_is_gpu_buffer_p(ptr);
}

int coll_adapt_cuda_get_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo)
{
    int i;
    int nb_gpus = 0;
    mca_common_cuda_get_device(&(gpu_topo->gpu_id));
    mca_common_cuda_get_device_count(&nb_gpus);
    int *gpu_numa = (int *)malloc(sizeof(int) * nb_gpus * 2);
    for (i = 0; i < nb_gpus; i++) {
        gpu_numa[i] = i;
    }
    int start_gpu = 0;
    gpu_numa[nb_gpus + start_gpu] = 0;
    int peer_access = 0;
    for (i = 1; i < nb_gpus; i++) {
        mca_common_cuda_device_can_access_peer(&peer_access, start_gpu, i);
        if (1 == peer_access) {
            gpu_numa[nb_gpus + i] = gpu_numa[nb_gpus + start_gpu];
        } else {
            gpu_numa[nb_gpus + i] = gpu_numa[nb_gpus + start_gpu] + 1;
            start_gpu = i;
        }
    }
    gpu_topo->nb_gpus = nb_gpus;
    gpu_topo->gpu_numa = gpu_numa;
    
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_free_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo)
{
    free(gpu_topo->gpu_numa);
    return OMPI_SUCCESS;
}