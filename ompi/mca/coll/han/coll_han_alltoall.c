/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Bull S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

void mac_coll_han_set_alltoall_argu(mca_alltoall_argu_t *argu,
                                    mca_coll_task_t *cur_task,
                                    void *sbuf,
                                    void *sbuf_inter_free,
                                    int scount,
                                    struct ompi_datatype_t *sdtype,
                                    void *rbuf,
                                    int rcount,
                                    struct ompi_datatype_t *rdtype,
                                    int root_low_rank,
                                    struct ompi_communicator_t *comm,
                                    struct ompi_communicator_t *up_comm,
                                    struct ompi_communicator_t *low_comm,
                                    int w_rank,
                                    int w_size,
                                    bool noop,
                                    bool is_mapbycore,
                                    int *topo,
                                    ompi_request_t *req)
{
    argu->cur_task = cur_task;
    argu->sbuf = sbuf;
    argu->sbuf_inter_free = sbuf_inter_free;
    argu->scount = scount;
    argu->sdtype = sdtype;
    argu->rbuf = rbuf;
    argu->rcount = rcount;
    argu->rdtype = rdtype;
    argu->root_low_rank = root_low_rank;
    argu->comm = comm;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->w_rank = w_rank;
    argu->w_size = w_size;
    argu->noop = noop;
    argu->is_mapbycore = is_mapbycore;
    argu->topo = topo;
    argu->req = req;
}

int mca_coll_han_alltoall_intra(const void *sbuf, int scount,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf, int rcount,
                                struct ompi_datatype_t *rdtype,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{

    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[INTRA_NODE];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);
    int low_rank = ompi_comm_rank(low_comm);

    ompi_request_t *temp_request = NULL;
    /* Set up request */
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = han_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;

    /* Init topo */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);

    int root_low_rank = 0;
    /* Create lg (lower level gather) task */
    mca_coll_task_t *lg = OBJ_NEW(mca_coll_task_t);
    /* Setup alltoall arguments */
    mca_alltoall_argu_t *ata_argu = malloc(sizeof(mca_alltoall_argu_t));
    mac_coll_han_set_alltoall_argu(ata_argu, lg, (char *)sbuf, NULL, scount, sdtype,
                                   (char *)rbuf, rcount, rdtype,
                                   root_low_rank,
                                   comm, up_comm, low_comm,
                                   w_rank, w_size,
                                   low_rank != root_low_rank,
                                   han_module->is_mapbycore, topo,
                                   temp_request);
    /* Init lg task */
    init_task(lg, mca_coll_han_alltoall_lg_task, (void *)(ata_argu));
    /* Issure lg task */
    issue_task(lg);

    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);

    return OMPI_SUCCESS;
}

/* lg: lower level (shared memory) gather task */
int mca_coll_han_alltoall_lg_task(void *task_argu)
{
    int i;
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Alltoall:  lg\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);

    int low_size = ompi_comm_size(t->low_comm);

    /* If the process is one of the node leader */
    /* allocate the intermediary sending buffer
     * to perform gather on leaders on the low sub communicator */
    char *tmp_sbuf = NULL;
    char *tmp_sbuf_start = NULL;
    if (!t->noop){
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        ssize = opal_datatype_span(&t->sdtype->super,
                                   (int64_t)t->scount * low_size * t->w_size, &sgap);
        /* intermediary buffer on node leaders to gather low data */
        tmp_sbuf = (char *)malloc(ssize);
        tmp_sbuf_start = tmp_sbuf - sgap;
    }

    /* 1. low gather on node leaders into tmp_sbuf */
    t->low_comm->c_coll->coll_gather((char *)t->sbuf, t->scount * t->w_size, t->sdtype,
                                     tmp_sbuf_start, t->scount * t->w_size,
                                     t->sdtype, t->root_low_rank,
                                     t->low_comm, t->low_comm->c_coll->coll_gather_module);

    /* 2. reorder the node leader's into sbuf.
     * if ranks are not mapped in topological order, data needs to be reordered
     * (see reorder_gather)
     */
    char *reorder_sbuf = NULL;
    char *reorder_sbuf_start = NULL;
    if (!t->noop){
        /* allocate the intermediary buffer
         * to gather on leaders on the low sub communicator 
         */
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        ssize = opal_datatype_span(&t->sdtype->super,
                                   (int64_t)t->scount * low_size * t->w_size, &sgap);
        /* intermediary buffer on node leaders to reorder data */
        reorder_sbuf = (char *)malloc(ssize);
        reorder_sbuf_start = reorder_sbuf - sgap;

        ompi_coll_han_reorder_alltoall_scatter(tmp_sbuf_start,
                                               t->scount, t->sdtype,
                                               (char *)reorder_sbuf_start,
                                               t->scount, t->sdtype,
                                               t->comm, t->up_comm, t->low_comm,
                                               t->topo);
        if (tmp_sbuf != NULL){
            free(tmp_sbuf);
            tmp_sbuf = NULL;
            tmp_sbuf_start = NULL;
        }
    }

    t->sbuf = reorder_sbuf_start;
    t->sbuf_inter_free = reorder_sbuf;

    /* Create uata (upper level alltoall) task */
    mca_coll_task_t *uata = OBJ_NEW(mca_coll_task_t);
    /* Setup uata task arguments */
    t->cur_task = uata;
    /* Init uata task */
    init_task(uata, mca_coll_han_alltoall_uata_task, (void *)t);
    /* Issure uata task */
    issue_task(uata);

    return OMPI_SUCCESS;
}

/* uag: upper level (inter-node) all-gather task */
int mca_coll_han_alltoall_uata_task(void *task_argu)
{
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Alltoall:  uata\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);

    char *tmp_rbuf = NULL;
    char *tmp_rbuf_start = NULL;
    if (!t->noop){
        int low_size = ompi_comm_size(t->low_comm);
        int low_rank = ompi_comm_rank(t->low_comm);
        if (0 == low_rank){
            ptrdiff_t rsize, rgap = 0;
            /* Compute the size to receive all the local data, 
             * including datatypes empty gaps 
             */
            rsize = opal_datatype_span(&t->rdtype->super,
                                       (int64_t)t->rcount * low_size * t->w_size, &rgap);
            /* intermediary buffer on node leaders to perform alltoall on upper level */
            tmp_rbuf = (char *)malloc(rsize);
            tmp_rbuf_start = tmp_rbuf - rgap;
        }

        t->up_comm->c_coll->coll_alltoall(t->sbuf,
                                          t->scount * low_size * low_size, t->sdtype,
                                          tmp_rbuf_start,
                                          t->rcount * low_size * low_size, t->rdtype,
                                          t->up_comm, t->up_comm->c_coll->coll_alltoall_module);
        if (t->sbuf_inter_free != NULL){
            free(t->sbuf_inter_free);
            t->sbuf_inter_free = NULL;
            t->sbuf = NULL;
        }
    }

    t->sbuf = tmp_rbuf_start;
    t->sbuf_inter_free = tmp_rbuf;

    /* Create ls (low level scatter) task */
    mca_coll_task_t *ls = OBJ_NEW(mca_coll_task_t);
    /* Setup ls task arguments */
    t->cur_task = ls;
    /* Init ls task */
    init_task(ls, mca_coll_han_alltoall_ls_task, (void *)t);
    /* Issure ls task */
    issue_task(ls);

    return OMPI_SUCCESS;
}

/* ls: lower level (shared memory) scatter task */
int mca_coll_han_alltoall_ls_task(void *task_argu)
{
    int i;
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Alltoall:  ls\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);
    int low_size = ompi_comm_size(t->low_comm);
    int low_rank = ompi_comm_rank(t->low_comm);
    int up_size = ompi_comm_size(t->up_comm);

    /* 4. up scatter: leaders scatter data on their nodes */
    char *reorder_rbuf = NULL;
    char *reorder_rbuf_start = NULL;

    if (low_rank == t->root_low_rank){
        ptrdiff_t rsize, rgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        rsize = opal_datatype_span(&t->sdtype->super,
                                   (int64_t)t->rcount * low_size * t->w_size, &rgap);
        /* intermediary buffer on node leaders to reorder data */
        reorder_rbuf = (char *)malloc(rsize);
        reorder_rbuf_start = reorder_rbuf - rgap;

        ompi_coll_han_reorder_alltoall_gather(t->sbuf,
                                              (char *)reorder_rbuf_start,
                                              t->rcount, t->sdtype,
                                              t->comm, t->up_comm, t->low_comm,
                                              t->topo);
        if (t->sbuf_inter_free != NULL){
            free(t->sbuf_inter_free);
            t->sbuf_inter_free = NULL;
            t->sbuf = NULL;
        }
    }

    t->low_comm->c_coll->coll_scatter(reorder_rbuf_start,
                                      t->w_size * t->rcount, t->rdtype,
                                      (char *)t->rbuf,
                                      t->w_size * t->rcount, t->rdtype,
                                      t->root_low_rank, t->low_comm,
                                      t->low_comm->c_coll->coll_scatter_module);
    if (reorder_rbuf != NULL){
        free(reorder_rbuf);
        reorder_rbuf = NULL;
        reorder_rbuf_start = NULL;
    }

    ompi_request_t *temp_req = t->req;
    free(t);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;
}

int mca_coll_han_alltoall_intra_simple(const void *sbuf, int scount,
                                       struct ompi_datatype_t *sdtype,
                                       void *rbuf, int rcount,
                                       struct ompi_datatype_t *rdtype,
                                       struct ompi_communicator_t *comm,
                                       mca_coll_base_module_t *module)
{
    int i;
    int root_low_rank = 0;

    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[INTRA_NODE];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* discovery topology */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced){
        return han_module->previous_alltoall(sbuf, scount, sdtype,
                                             rbuf, rcount, rdtype,
                                             comm, han_module->previous_alltoall_module);
    }

    /* setup up/low and global coordinates */
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int up_rank = ompi_comm_rank(up_comm);
    int up_size = ompi_comm_size(up_comm);
    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);

    /* allocate the intermediary sending buffer
     * to perform gather on leaders on the low sub communicator */
    char *tmp_sbuf = NULL;
    char *tmp_sbuf_start = NULL;
    if (low_rank == root_low_rank){
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        ssize = opal_datatype_span(&sdtype->super,
                                   (int64_t)scount * low_size * w_size, &sgap);
        /* intermediary buffer on node leaders to gather on low comm */
        tmp_sbuf = (char *)malloc(ssize);
        tmp_sbuf_start = tmp_sbuf - sgap;
    }

    /* 1. low gather on node leaders into tmp_sbuf */
    low_comm->c_coll->coll_gather((char *)sbuf, scount * w_size, sdtype,
                                  tmp_sbuf_start, scount * w_size, sdtype,
                                  root_low_rank, low_comm,
                                  low_comm->c_coll->coll_gather_module);

    /* 2. reorder in the node leaders. */
    char *reorder_sbuf = NULL;
    char *reorder_sbuf_start = NULL;

    if (low_rank == root_low_rank){
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        ssize = opal_datatype_span(&sdtype->super,
                                   (int64_t)scount * low_size * w_size, &sgap);
        /* intermediary buffer on node leaders to reorder on low comm root */
        reorder_sbuf = (char *)malloc(ssize);
        reorder_sbuf_start = reorder_sbuf - sgap;

        ompi_coll_han_reorder_alltoall_scatter(tmp_sbuf_start, scount, sdtype,
                                               reorder_sbuf_start,
                                               scount, sdtype,
                                               comm, up_comm, low_comm, topo);
        if (tmp_sbuf != NULL){
            free(tmp_sbuf);
            tmp_sbuf = NULL;
            tmp_sbuf_start = NULL;
        }
    }

    /* 3. alltoall between node leaders */
    char *tmp_rbuf = NULL;
    char *tmp_rbuf_start = NULL;

    if (low_rank == root_low_rank){
        ptrdiff_t rsize, rgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        rsize = opal_datatype_span(&rdtype->super,
                                   (int64_t)rcount * low_size * w_size, &rgap);
        /* intermediary buffer on node leaders to perform alltoall operation */
        tmp_rbuf = (char *)malloc(rsize);
        tmp_rbuf_start = tmp_rbuf - rgap;

        up_comm->c_coll->coll_alltoall(reorder_sbuf_start,
                                       scount * low_size * low_size, sdtype,
                                       tmp_rbuf_start,
                                       rcount * low_size * low_size, rdtype,
                                       up_comm,
                                       up_comm->c_coll->coll_alltoall_module);
        if (reorder_sbuf != NULL){
            free(reorder_sbuf);
            reorder_sbuf = NULL;
            reorder_sbuf_start = NULL;
        }
    }

    /* 4. reorder: leaders reorder data to prepare it 
     * to be scattered on their nodes 
     */
    char *reorder_rbuf = NULL;
    char *reorder_rbuf_start = NULL;

    if (low_rank == root_low_rank){
        ptrdiff_t rsize, rgap = 0;
        /* Compute the size to receive all the local data, 
         * including datatypes empty gaps 
         */
        rsize = opal_datatype_span(&sdtype->super,
                                   (int64_t)rcount * low_size * w_size, &rgap);
        /* intermediary buffer on node leaders to reorder data */
        reorder_rbuf = (char *)malloc(rsize);
        reorder_rbuf_start = reorder_rbuf - rgap;

        ompi_coll_han_reorder_alltoall_gather(tmp_rbuf_start,
                                              reorder_rbuf_start,
                                              rcount, sdtype,
                                              comm, up_comm, low_comm, topo);
        if (tmp_rbuf != NULL){
            free(tmp_rbuf);
            tmp_rbuf = NULL;
            tmp_rbuf_start = NULL;
        }
    }

    /* 4. low scatter: leaders scatter data on their nodes */
    low_comm->c_coll->coll_scatter(reorder_rbuf_start, w_size * rcount, rdtype,
                                   (char *)rbuf, w_size * rcount, rdtype,
                                   root_low_rank, low_comm,
                                   low_comm->c_coll->coll_scatter_module);

    if (reorder_rbuf != NULL){
        free(reorder_rbuf);
        reorder_rbuf = NULL;
        reorder_rbuf_start = NULL;
    }

    return OMPI_SUCCESS;
}

/* Reorder after gather operation */
void ompi_coll_han_reorder_alltoall_gather(const void *sbuf,
                                           void *rbuf, int rcount,
                                           struct ompi_datatype_t *rdtype,
                                           struct ompi_communicator_t *comm,
                                           struct ompi_communicator_t *up_comm,
                                           struct ompi_communicator_t *low_comm,
                                           int *topo)
{
    int i, j, k, pos;
    int topolevel = 2; // always 2 levels in topo
    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);
    int up_size = ompi_comm_size(up_comm);
    int low_size = ompi_comm_size(low_comm);
    ptrdiff_t rextent;
    ompi_datatype_type_extent(rdtype, &rextent);
    for (j = 0; j < low_size; j++){
        for (k = 0; k < up_size; k++){
            for (i = 0; i < low_size; i++){
                ptrdiff_t block_size = rextent * (ptrdiff_t)rcount;
                pos = i + k * low_size;
                OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                     "[%d]: Future reorder from %d to %d\n",
                                     w_rank, pos * topolevel + 1,
                                     topo[pos * topolevel + 1]));
                ptrdiff_t src_shift = block_size * (pos + k * low_size * (low_size - 1) +
                                                    j * low_size);
                ptrdiff_t dest_shift = block_size * (ptrdiff_t)topo[pos * topolevel + 1] +
                                       j * w_size * block_size;
                ompi_datatype_copy_content_same_ddt(rdtype,
                                                    (ptrdiff_t)rcount,
                                                    (char *)rbuf + dest_shift,
                                                    (char *)sbuf + src_shift);
            }
        }
    }
}

/* Reorder sbuf based on rank.*/
void ompi_coll_han_reorder_alltoall_scatter(const void *sbuf,
                                            int scount,
                                            struct ompi_datatype_t *sdtype,
                                            const void *rbuf,
                                            int rcount,
                                            struct ompi_datatype_t *rdtype,
                                            struct ompi_communicator_t *comm,
                                            struct ompi_communicator_t *up_comm,
                                            struct ompi_communicator_t *low_comm,
                                            int *topo)
{
    int i, j, k;

    /* discovery topology */
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);
    int w_size = ompi_comm_size(comm);

    /* Reorder */
    ptrdiff_t sextent;
    ompi_datatype_type_extent(sdtype, &sextent);
    for (k = 0; k < low_size; k++){
        for (i = 0; i < up_size; i++){
            for (j = 0; j < low_size; j++){
                ompi_datatype_copy_content_same_ddt(sdtype, (ptrdiff_t)scount,
                                                    (char *)rbuf + k * scount +
                                                        sextent * (i * low_size + j) * low_size * (ptrdiff_t)scount,
                                                    (char *)sbuf + w_size * scount * k +
                                                        sextent * (ptrdiff_t)topo[(i * low_size + j) * 2 + 1] * (ptrdiff_t)scount);
            }
        }
    }
}

/* Alltoall variant with loop */


int
mca_coll_han_alltoall_loop_intra(const void *sbuf, int scount,
                             struct ompi_datatype_t *sdtype,
                             void *rbuf, int rcount,
                             struct ompi_datatype_t *rdtype,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t * module)
{
    int w_rank;
    int w_size;
    w_rank = ompi_comm_rank(comm);
    w_size = ompi_comm_size(comm);

    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;
    mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[INTRA_NODE];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int low_rank = ompi_comm_rank(low_comm);

    ompi_request_t *temp_request = NULL;
    /* Set up request */
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = han_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;

    /* Init topo */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);

    int root_low_rank = 0;
    /* Create lg (lower level gather) task */
    mca_coll_task_t *lg = OBJ_NEW(mca_coll_task_t);
    /* Setup alltoall arguments */
    mca_alltoall_argu_t *ata_argu = malloc(sizeof(mca_alltoall_argu_t));
    mac_coll_han_set_alltoall_argu(ata_argu, lg, (char *) sbuf, NULL, scount, sdtype, rbuf, rcount,
                                    rdtype, root_low_rank, comm, up_comm, low_comm, w_rank, w_size,
                                    low_rank != root_low_rank, han_module->is_mapbycore, topo,
                                    temp_request);
    /* Init lg task */
    init_task(lg, mca_coll_han_alltoall_loop_lg_task, (void *) (ata_argu));
    /* Issure lg task */
    issue_task(lg);

    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);

    return OMPI_SUCCESS;
}

/* lg: lower level (shared memory) gather task */
int mca_coll_han_alltoall_loop_lg_task(void *task_argu)
{
    int i;
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *) task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] HAN Alltoall:  lg\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);

    /* If the process is one of the node leader */
    /* allocate the intermediary sending buffer
     * to perform gather on leaders on the low sub communicator */
    char *tmp_sbuf = NULL;
    char *tmp_sbuf_start = NULL;
    int low_size = ompi_comm_size(t->low_comm);
    if (!t->noop) {
        ptrdiff_t ssize, sgap = 0;
        ssize = opal_datatype_span(&t->sdtype->super, 
                                   (int64_t) t->scount * low_size * t->w_size, &sgap);
        tmp_sbuf = (char *) malloc(ssize);
        tmp_sbuf_start = tmp_sbuf - sgap;
    }
    /* 1. low gather on node leaders into tmp_sbuf */
    for (i = 0; i < t->w_size; i++){
        ptrdiff_t src_shift = i * t->scount;
        ptrdiff_t dest_shift = i * low_size * t->scount;                                      
        t->low_comm->c_coll->coll_gather((char *)t->sbuf + src_shift, 
                                         t->scount, t->sdtype,
                                         tmp_sbuf_start + dest_shift, 
                                         t->scount, t->sdtype, 
                                         t->root_low_rank,
                                         t->low_comm, 
                                         t->low_comm->c_coll->coll_gather_module);    
    }

    /* 2. reorder the node leader's into sbuf.
     * if ranks are not mapped in topological order, data needs to be reordered
     * (see reorder_gather)
     */
    char *reorder_sbuf = NULL;
    char *reorder_sbuf_start = NULL;
    if (!t->noop) {
        /* allocate the intermediary buffer
         * to gather on leaders on the low sub communicator */
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, including datatypes empty gaps */
        ssize = opal_datatype_span(&t->sdtype->super, 
                                   (int64_t)t->scount * low_size * t->w_size, &sgap);
        /* intermediary buffer on node leaders to gather on low comm */
        reorder_sbuf = (char *) malloc(ssize);
        reorder_sbuf_start = reorder_sbuf - sgap;
    
        if (t->is_mapbycore) {
            reorder_sbuf_start = tmp_sbuf_start;
        } else {
            ompi_coll_han_reorder_scatter(tmp_sbuf_start,
                                          t->scount * low_size, t->sdtype, 
                                          (char*)reorder_sbuf_start,
                                          t->scount * low_size, t->sdtype,
                                          t->up_comm,t->low_comm,t->topo);
                                          
            if (tmp_sbuf != NULL) {
                free(tmp_sbuf);
                tmp_sbuf = NULL;
                tmp_sbuf_start = NULL;
            }
        }
    }

    t->sbuf = reorder_sbuf_start;
    t->sbuf_inter_free = reorder_sbuf;

    /* Create uata (upper level alltoall) task */
    mca_coll_task_t *uata = OBJ_NEW(mca_coll_task_t);
    /* Setup uag task arguments */
    t->cur_task = uata;
    /* Init uag task */
    init_task(uata, mca_coll_han_alltoall_loop_uata_task, (void *) t);
    /* Issure uag task */
    issue_task(uata);

    return OMPI_SUCCESS;
}

/* uag: upper level (inter-node) all-gather task */
int mca_coll_han_alltoall_loop_uata_task(void *task_argu)
{
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *) task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] HAN Alltoall:  uata\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);

    char *tmp_rbuf = NULL;
    char *tmp_rbuf_start = NULL;
    if (!t->noop){
        int low_size = ompi_comm_size(t->low_comm);
        int low_rank = ompi_comm_rank(t->low_comm);
        if (0 == low_rank){
            ptrdiff_t rsize, rgap = 0;
            rsize = opal_datatype_span(&t->rdtype->super, 
                                       (int64_t)t->rcount * low_size * t->w_size, &rgap);
            tmp_rbuf = (char *) malloc(rsize);
            tmp_rbuf_start = tmp_rbuf - rgap; 
        }

        t->up_comm->c_coll->coll_alltoall(t->sbuf, 
                                          t->scount * low_size * low_size, 
                                          t->sdtype,
                                          tmp_rbuf_start, 
                                          t->rcount * low_size * low_size, 
                                          t->rdtype,
                                          t->up_comm, 
                                          t->up_comm->c_coll->coll_alltoall_module);
        if (t->sbuf_inter_free != NULL) {
            free(t->sbuf_inter_free);
            t->sbuf_inter_free = NULL;
            t->sbuf = NULL;
        }
    }

    t->sbuf = tmp_rbuf_start;
    t->sbuf_inter_free = tmp_rbuf;

    /* Create ls (low level scatter) task */
    mca_coll_task_t *ls = OBJ_NEW(mca_coll_task_t);
    /* Setup ls task arguments */
    t->cur_task = ls;
    /* Init ls task */
    init_task(ls, mca_coll_han_alltoall_loop_ls_task, (void *) t);
    /* Issure lb task */
    issue_task(ls);
    
    return OMPI_SUCCESS;
}

/* ls: lower level (shared memory) scatter task */
int mca_coll_han_alltoall_loop_ls_task(void *task_argu)
{
    int i;
    mca_alltoall_argu_t *t = (mca_alltoall_argu_t *) task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] HAN Alltoall:  ls\n",
                     t->w_rank));
    OBJ_RELEASE(t->cur_task);
    int low_size = ompi_comm_size(t->low_comm);
    int up_size = ompi_comm_size(t->up_comm);

    /* 4. up scatter: leaders scatter data on their nodes */
    char *reorder_rbuf = NULL;
    char *reorder_rbuf_start = NULL;
    if (t->is_mapbycore) {
        reorder_rbuf_start = (char *)t->rbuf;
    } else {
        ptrdiff_t rsize, rgap = 0;
        rsize = opal_datatype_span(&t->rdtype->super, 
                                   (int64_t)t->rcount * t->w_size, &rgap);
        reorder_rbuf = (char *) malloc(rsize);
        reorder_rbuf_start = reorder_rbuf - rgap;
    }

    for (i = 0; i < up_size; i++){
        ptrdiff_t src_shift = i * low_size * low_size * t->rcount;
        ptrdiff_t dest_shift = i * low_size * t->rcount;  
        t->low_comm->c_coll->coll_scatter(t->sbuf + src_shift, 
                                          low_size * t->rcount, t->rdtype,
                                          reorder_rbuf_start + dest_shift, 
                                          low_size * t->rcount, t->rdtype,
                                          t->root_low_rank, t->low_comm,
                                          t->low_comm->c_coll->coll_scatter_module);
    }

   if (!t->is_mapbycore) {
        ompi_coll_han_reorder_gather(reorder_rbuf_start,
                                    (char *)t->rbuf, t->rcount , t->rdtype,
                                    t->comm, t->topo);
        free(reorder_rbuf);
        reorder_rbuf = NULL;
    }

    ompi_request_t *temp_req = t->req;
    free(t);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;
}

int
mca_coll_han_alltoall_loop_intra_simple(const void *sbuf, int scount,
                                    struct ompi_datatype_t *sdtype,
                                    void* rbuf, int rcount,
                                    struct ompi_datatype_t *rdtype,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module)
{
    int i;

    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[INTRA_NODE];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* discovery topology */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced){
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                           "han cannot handle alltoall with this communicator. It need to fall back on another component\n"));
        return han_module->previous_alltoall(sbuf, scount, sdtype, rbuf,
                                            rcount, rdtype,
                                            comm, han_module->previous_alltoall_module);
    }

    /* setup up/low and global coordinates */
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int up_rank = ompi_comm_rank(up_comm);
    int up_size = ompi_comm_size(up_comm);
    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);
    int root_low_rank = 0; 

    /* allocate the intermediary sending buffer
     * to perform gather on leaders on the low sub communicator */
    char *tmp_sbuf = NULL;
    char *tmp_sbuf_start = NULL;
    if (low_rank == root_low_rank) {
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, including datatypes empty gaps */
        ssize = opal_datatype_span(&sdtype->super, 
                                   (int64_t)scount * low_size * w_size, &sgap);
        /* intermediary buffer on node leaders to gather on low comm */
        tmp_sbuf = (char *) malloc(ssize);
        tmp_sbuf_start = tmp_sbuf - sgap;
    }

    /* 1. low gather on node leaders into tmp_sbuf */
    for (i = 0; i < w_size; i++){
        ptrdiff_t src_shift = i * scount;
        ptrdiff_t dest_shift = i * low_size * scount;                                      
        low_comm->c_coll->coll_gather((char *)sbuf + src_shift, 
                                      scount, sdtype,
                                      (char *)tmp_sbuf_start + dest_shift, 
                                      scount, sdtype, 
                                      root_low_rank,
                                      low_comm, 
                                      low_comm->c_coll->coll_gather_module);    
    }

    /* 2. reorder the node leader's into sbuf.
     * if ranks are not mapped in topological order, data needs to be reordered
     * (see reorder_gather)
     */
 
    /* allocate the intermediary buffer
     * to gather on leaders on the low sub communicator */
    char *reorder_sbuf = NULL;
    char *reorder_sbuf_start = NULL;

    if (low_rank == root_low_rank) {
        ptrdiff_t ssize, sgap = 0;
        /* Compute the size to receive all the local data, including datatypes empty gaps */
        ssize = opal_datatype_span(&sdtype->super, 
                                   (int64_t)scount * low_size * w_size, &sgap);
        /* intermediary buffer on node leaders to gather on low comm */
        reorder_sbuf = (char *) malloc(ssize);
        reorder_sbuf_start = reorder_sbuf - sgap;
    
        if (han_module->is_mapbycore) {
            reorder_sbuf_start = tmp_sbuf_start;
        } else {
            ompi_coll_han_reorder_scatter(tmp_sbuf_start,scount * low_size, sdtype, 
                                          (char*)reorder_sbuf_start,scount * low_size, sdtype,
                                          up_comm,low_comm, topo);
            if (tmp_sbuf != NULL) {
                free(tmp_sbuf);
                tmp_sbuf = NULL;
                tmp_sbuf_start = NULL;
            }
        }
    }

    /* 3. alltoall between node leaders, from tmp_sbuf to tmp_rbuf */
    char *tmp_rbuf = NULL;
    char *tmp_rbuf_start = NULL;

    if (low_rank == root_low_rank) {
        ptrdiff_t rsize, rgap = 0;
        rsize = opal_datatype_span(&rdtype->super, 
                                   (int64_t)rcount * low_size * w_size, &rgap);
        tmp_rbuf = (char *) malloc(rsize);
        tmp_rbuf_start = tmp_rbuf - rgap; 
        
        up_comm->c_coll->coll_alltoall(reorder_sbuf_start, 
                                       scount * low_size * low_size, sdtype,
                                       tmp_rbuf_start, 
                                       rcount * low_size * low_size, rdtype,
                                       up_comm, up_comm->c_coll->coll_alltoall_module);
        if (reorder_sbuf != NULL) {
            free(reorder_sbuf);
            reorder_sbuf = NULL;
            reorder_sbuf_start = NULL;
        }
    }

    /* 4. up scatter: leaders scatter data on their nodes */
    char *reorder_rbuf = NULL;
    char *reorder_rbuf_start = NULL;
    if (han_module->is_mapbycore) {
        reorder_rbuf_start = (char *)rbuf;
    } else {
        ptrdiff_t rsize, rgap = 0;
        rsize = opal_datatype_span(&rdtype->super, 
                                   (int64_t)rcount * w_size, &rgap);
        reorder_rbuf = (char *) malloc(rsize);
        reorder_rbuf_start = reorder_rbuf - rgap;
    }

    for (i = 0; i < up_size; i++){
        ptrdiff_t src_shift = i * low_size * low_size * rcount;
        ptrdiff_t dest_shift = i * low_size * rcount;  
        low_comm->c_coll->coll_scatter(tmp_rbuf_start + src_shift, 
                                       low_size * rcount, rdtype,
                                       reorder_rbuf_start + dest_shift, 
                                       low_size * rcount, rdtype,
                                       root_low_rank, low_comm,
                                       low_comm->c_coll->coll_scatter_module);
    }

   if (!han_module->is_mapbycore) {
        ompi_coll_han_reorder_gather(reorder_rbuf_start,
                                    (char *)rbuf, rcount , rdtype,
                                    comm, topo);
        free(reorder_rbuf);
        reorder_rbuf = NULL;
    }

    return OMPI_SUCCESS;
}
