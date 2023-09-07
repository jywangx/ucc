/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "core/ucc_team.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>

enum sharp_datatype ucc_to_sharp_dtype[] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)]            = SHARP_DTYPE_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)]            = SHARP_DTYPE_INT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)]            = SHARP_DTYPE_LONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT128)]           = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)]           = SHARP_DTYPE_UNSIGNED_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)]           = SHARP_DTYPE_UNSIGNED,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)]           = SHARP_DTYPE_UNSIGNED_LONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT128)]          = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)]          = SHARP_DTYPE_FLOAT_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)]          = SHARP_DTYPE_FLOAT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)]          = SHARP_DTYPE_DOUBLE,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128)]         = SHARP_DTYPE_NULL,
#if SHARP_API > SHARP_VERSION(3, 0)
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]             = SHARP_DTYPE_UNKNOWN,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]            = SHARP_DTYPE_UNKNOWN,
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)]         = SHARP_DTYPE_UNKNOWN,
#else
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]             = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]            = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)]         = SHARP_DTYPE_NULL,
#endif
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32_COMPLEX)]  = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64_COMPLEX)]  = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128_COMPLEX)] = SHARP_DTYPE_NULL,
};

enum sharp_reduce_op ucc_to_sharp_reduce_op[] = {
    [UCC_OP_SUM]         = SHARP_OP_SUM,
    [UCC_OP_PROD]        = SHARP_OP_NULL,
    [UCC_OP_MAX]         = SHARP_OP_MAX,
    [UCC_OP_MIN]         = SHARP_OP_MIN,
    [UCC_OP_LAND]        = SHARP_OP_LAND,
    [UCC_OP_LOR]         = SHARP_OP_LOR,
    [UCC_OP_LXOR]        = SHARP_OP_LXOR,
    [UCC_OP_BAND]        = SHARP_OP_BAND,
    [UCC_OP_BOR]         = SHARP_OP_BOR,
    [UCC_OP_BXOR]        = SHARP_OP_BXOR,
    [UCC_OP_MAXLOC]      = SHARP_OP_MAXLOC,
    [UCC_OP_MINLOC]      = SHARP_OP_MINLOC,
    [UCC_OP_AVG]         = SHARP_OP_NULL,
};

enum sharp_data_memory_type ucc_to_sharp_memtype[] = {
    [UCC_MEMORY_TYPE_HOST]         = SHARP_MEM_TYPE_HOST,
    [UCC_MEMORY_TYPE_CUDA]         = SHARP_MEM_TYPE_CUDA,
    [UCC_MEMORY_TYPE_CUDA_MANAGED] = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_ROCM]         = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_ROCM_MANAGED] = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_LAST]         = SHARP_MEM_TYPE_LAST,
};

static inline ucc_status_t ucc_tl_sharp_status_to_ucc(int status)
{
    switch (status) {
    case SHARP_COLL_SUCCESS:
        return UCC_OK;
    case SHARP_COLL_ENOMEM:
        return UCC_ERR_NO_MEMORY;
    case SHARP_COLL_ENOT_SUPP:
        return UCC_ERR_NOT_SUPPORTED;
    case SHARP_COLL_EINVAL:
        return UCC_ERR_INVALID_PARAM;
    case SHARP_COLL_ENO_RESOURCE:
        return UCC_ERR_NO_RESOURCE;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

static ucc_tl_sharp_reg_t ucc_tl_sharp_reg_null = { .mr = NULL };

static ucc_status_t
ucc_tl_sharp_mem_register(ucc_tl_sharp_context_t *ctx, ucc_tl_sharp_team_t *team,
                          void *addr, size_t length, ucc_tl_sharp_reg_t **reg)
{
    ucc_rcache_region_t          *rregion;
    ucc_tl_sharp_rcache_region_t *region;
    ucc_status_t                  status;
    ucc_tl_sharp_reg_t           *r;
    ucc_rcache_t                 *rcache;
    struct sharp_coll_context    *sharp_ctx;

    if (length < ctx->cfg.reg_threshold) {
        *reg = &ucc_tl_sharp_reg_null;
        return UCC_OK;
    }

    sharp_ctx = team->sharp_context;
    rcache    = team->rcache;

    if (rcache) {
        status = ucc_rcache_get(rcache, (void *)addr, length, NULL,
                                &rregion);
        if (status != UCC_OK) {
            tl_error(ctx->super.super.lib, "ucc_rcache_get failed");
            return UCC_ERR_INVALID_PARAM;
        }
        region = ucc_derived_of(rregion, ucc_tl_sharp_rcache_region_t);
        *reg   = &region->reg;
    } else {
        r = ucc_malloc(sizeof(ucc_tl_sharp_reg_t), "sharp reg");
        if (!r) {
            tl_error(ctx->super.super.lib, "failed to allocate reg data");
            return UCC_ERR_NO_MEMORY;
        }

        sharp_coll_reg_mr(sharp_ctx, addr, length, &r->mr);
        *reg = r;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_sharp_mem_deregister(ucc_tl_sharp_team_t *team, ucc_tl_sharp_reg_t *reg)
{
    ucc_tl_sharp_rcache_region_t *region;
    ucc_rcache_t *rcache;
    struct       sharp_coll_context *sharp_ctx;

    if (reg == &ucc_tl_sharp_reg_null) {
        return UCC_OK;
    }

    sharp_ctx = team->sharp_context;
    rcache    = team->rcache;

    if (rcache) {
        region = ucc_container_of(reg, ucc_tl_sharp_rcache_region_t, reg);
        ucc_rcache_region_put(rcache, &region->super);
    } else {
        sharp_coll_dereg_mr(sharp_ctx, reg->mr);
        ucc_free(reg);
    }

    return UCC_OK;
}

void ucc_tl_sharp_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_status_t         status  = UCC_OK;
    ucc_tl_sharp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_rank_t           rank    = task->super.team->params.rank;
    ucc_rank_t           size    = task->super.team->params.size;
    ucc_rank_t           root    = task->super.bargs.args.root;
    ucc_count_t          rcount  = task->super.bargs.args.dst.info.count;
    ucc_datatype_t       dt      = task->super.bargs.args.dst.info.datatype;
    void                *dst_buf = task->super.bargs.args.dst.info.buffer;
    void                *src_buf;
    size_t               rdata_size;
    int completed;

    if (task->req_handle != NULL) {
        completed = sharp_coll_req_test(task->req_handle);
        if (completed) {
            if (TASK_ARGS(task).coll_type == UCC_COLL_TYPE_ALLREDUCE) {
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->allreduce.s_mem_h);
                }
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->allreduce.r_mem_h);
            }
            if (TASK_ARGS(task).coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->reduce_scatter.s_mem_h);
                } else {
                    rcount /= size;
                }
                // printf("rank %d deregister %p\n", rank, task->reduce_scatter.r_mem_h->mr);
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->reduce_scatter.r_mem_h);
                rdata_size = ucc_dt_size(dt) * rcount;
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    src_buf = PTR_OFFSET(
                        task->reduce_scatter.scratch, 
                        rdata_size * rank
                    );
                } else {
                    src_buf = PTR_OFFSET(
                        task->super.bargs.args.dst.info.buffer, 
                        rdata_size * rank
                    );
                }
                // printf("rank %d copy %p to %p\n", rank, src_buf, dst_buf);
                if (src_buf != dst_buf) status = ucc_mc_memcpy(
                    dst_buf, src_buf, rdata_size, 
                    UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_HOST
                );
                if (status != UCC_OK) {
                    tl_error(
                        task->super.team->context->lib,
                        "failed to ucc_mc_memcpy for sharp reduce-scatter"
                    );
                    coll_task->status = status;
                }
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_mc_free(task->reduce_scatter.scratch_mc_header);
                }
            }
            if (TASK_ARGS(task).coll_type == UCC_COLL_TYPE_REDUCE) {
                if (UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->allreduce.r_mem_h);
                } else {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->allreduce.s_mem_h);
                    if (rank == root) {
                        ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                    task->allreduce.r_mem_h);
                    }
                }
            }
            if (TASK_ARGS(task).coll_type == UCC_COLL_TYPE_BCAST) {
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->bcast.mem_h);
            }
            sharp_coll_req_free(task->req_handle);
            coll_task->status = UCC_OK;
            UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                               "sharp_collective_done", 0);
        }
    }
}

ucc_status_t ucc_tl_sharp_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    int                  ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_barrier_start", 0);

    ret = sharp_coll_do_barrier_nb(team->sharp_comm, &task->req_handle);
    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "sharp_coll_do_barrier_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t          *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team  = TASK_TEAM(task);
    ucc_coll_args_t              *args  = &TASK_ARGS(task);
    size_t                        count = args->dst.info.count;
    ucc_datatype_t                dt    = args->dst.info.datatype;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        data_size;
    int                           ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_allreduce_start", 0);

    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];
    data_size  = ucc_dt_size(dt) * count;

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,data_size,
                                  &task->allreduce.s_mem_h);
    }
    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer, data_size,
                              &task->allreduce.r_mem_h);

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr        = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        reduce_spec.sbuf_desc.buffer.ptr        = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    }

    reduce_spec.sbuf_desc.buffer.length     = data_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    reduce_spec.rbuf_desc.buffer.length     = data_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec.length                      = count;
    reduce_spec.dtype                       = sharp_type;
    reduce_spec.op                          = op_type;

    ret = sharp_coll_do_allreduce_nb(team->sharp_comm, &reduce_spec, &task->req_handle);
    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "sharp_coll_do_allreduce_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

/**
 * Implemented by n parallel non-blocking reduce
*/
void ucc_tl_sharp_reduce_scatter_nr_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    int completed;
    int size = (int)(coll_task->bargs.team->size);

    //multiple reduce nb
    void ** request_list = (void **)task->reduce_scatter.reqs;
    for(int i = 0; i < size; i++){

        //check i th reduce_nb request
        completed = sharp_coll_req_test(request_list[i]);
        if(completed)
            continue;
        else
            return;//one of the request is not completed, return immediately
    }
    
    for(int i = 0; i < size; i++){//free all requests
        sharp_coll_req_free(request_list[i]);
    }

    coll_task->status = UCC_OK;
    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                        "sharp_collective_done", 0);
}

/**
 * Implemented by n parallel non-blocking reduce
*/
ucc_status_t ucc_tl_sharp_reduce_scatter_nr_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t          *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team  = TASK_TEAM(task);
    ucc_coll_args_t              *args  = &TASK_ARGS(task);
    size_t                        count = args->dst.info.count;
    ucc_datatype_t                dt    = args->dst.info.datatype;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        data_size;
    int                           ret;

    //int              rank = (int)(coll_task->bargs.team->rank);
    int              size = (int)(coll_task->bargs.team->size);

    //initialize sharp_req hands
    void **sharp_reqs;
    sharp_reqs = malloc(sizeof(void *)*size);

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_reduce_scatter_start", 0); // Not sure

    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];
    data_size  = ucc_dt_size(dt) * count;

    /*offset for each scatter*/
    long long offset = ((long long) data_size)/size;

    if(offset < 16 * 1024){
        //message size too small, not supported
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,data_size,
                                  &task->allreduce.s_mem_h);
    }
    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer, data_size,
                              &task->allreduce.r_mem_h);

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr        = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        reduce_spec.sbuf_desc.buffer.ptr        = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    }

    reduce_spec.sbuf_desc.buffer.length     = data_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    reduce_spec.rbuf_desc.buffer.length     = data_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec.length                      = (count/size);//reducce scatter 0
    reduce_spec.dtype                       = sharp_type;
    reduce_spec.root                        = 0;
    reduce_spec.op                          = op_type;

    ret = SHARP_COLL_SUCCESS;

    //use reduce non blocking

    char *srcBufPtrInChar = (char *) args->src.info.buffer;
    char *dstBufPtrInChar = (char *) args->dst.info.buffer;

    printf("lunching reduce nb\n");

    for(int rankCnt = 0; rankCnt < size; rankCnt++){

        ret = sharp_coll_do_reduce_nb(team->sharp_comm, &reduce_spec, &sharp_reqs[rankCnt]);

        /*update src and dst ptr*/
        srcBufPtrInChar += offset;
        reduce_spec.sbuf_desc.buffer.ptr  = (void *)srcBufPtrInChar;

        dstBufPtrInChar += offset;
        reduce_spec.rbuf_desc.buffer.ptr = (void *)dstBufPtrInChar;

        /*update root*/
        reduce_spec.root += 1;

    }

    //give the pointer of requestes list for later test
    task->reduce_scatter.reqs = (void *)sharp_reqs;

    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "reduce scatter REDUCENB failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

/**
 * Implemented by allreduce wrapped
*/
ucc_status_t ucc_tl_sharp_reduce_scatter_arw_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t          *task   = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team   = TASK_TEAM(task);
    ucc_coll_args_t              *args   = &TASK_ARGS(task);
    size_t                        scount = args->dst.info.count;
    ucc_datatype_t                dt     = args->dst.info.datatype;
    ucc_status_t                  status = UCC_OK;
    // ucc_rank_t                    rank   = team->super.super.params.rank;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        sdata_size;
    int                           ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_reduce_scatter_start", 0);

    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];

    if (UCC_IS_INPLACE(*args)) {
        sdata_size  = ucc_dt_size(dt) * scount;
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer, sdata_size,
                                  &task->reduce_scatter.r_mem_h);
        // printf("rank %d register %p\n", rank, task->reduce_scatter.r_mem_h->mr);
    } else {
        scount  = args->src.info.count;
        sdata_size  = ucc_dt_size(dt) * scount;
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer, sdata_size,
                                  &task->reduce_scatter.s_mem_h);
        // printf("rank %d register %p\n", rank, task->reduce_scatter.s_mem_h->mr);
        status = ucc_mc_alloc(&task->reduce_scatter.scratch_mc_header,
                              sdata_size,
                              args->dst.info.mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(team->super.super.context->lib,
                    "failed to allocate %zd bytes for reduce-scatter arw reduce_buf",
                    sdata_size);
            return status;
        }
        // printf("rank %d alloc %p\n", rank, task->reduce_scatter.scratch_mc_header->addr);
        task->reduce_scatter.scratch = task->reduce_scatter.scratch_mc_header->addr;
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, task->reduce_scatter.scratch, sdata_size,
                                  &task->reduce_scatter.r_mem_h);
        // printf("rank %d register %p\n", rank, task->reduce_scatter.r_mem_h->mr);
    }

    if (UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr        = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->reduce_scatter.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];

        reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    } else {
        reduce_spec.sbuf_desc.buffer.ptr        = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->reduce_scatter.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->src.info.mem_type];

        reduce_spec.rbuf_desc.buffer.ptr        = task->reduce_scatter.scratch;
    }

    reduce_spec.sbuf_desc.buffer.length     = sdata_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.length     = sdata_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->reduce_scatter.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec.length                      = scount;
    reduce_spec.dtype                       = sharp_type;
    reduce_spec.op                          = op_type;

    ret = sharp_coll_do_allreduce_nb(team->sharp_comm, &reduce_spec, &task->req_handle);
    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "sharp_coll_do_all_reduce_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *    args  = &TASK_ARGS(task);
    ucc_datatype_t       dt    = args->src.info.datatype;
    size_t               count = args->src.info.count;
    ucc_rank_t           root  = args->root;
    size_t               data_size;
    struct sharp_coll_bcast_spec bcast_spec;
    int                          ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_bcast_start", 0);

    data_size = ucc_dt_size(dt) * count;

    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,
                              data_size, &task->bcast.mem_h);

    bcast_spec.size                       = data_size;
    bcast_spec.root                       = root;
    bcast_spec.buf_desc.type              = SHARP_DATA_BUFFER;
    bcast_spec.buf_desc.buffer.ptr        = args->src.info.buffer;
    bcast_spec.buf_desc.buffer.length     = data_size;
    bcast_spec.buf_desc.buffer.mem_handle = task->bcast.mem_h->mr;
    bcast_spec.buf_desc.mem_type =
        ucc_to_sharp_memtype[args->src.info.mem_type];

    ret = sharp_coll_do_bcast_nb(team->sharp_comm, &bcast_spec,
                                 &task->req_handle);

    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        if (ret == SHARP_COLL_ENOT_SUPP) {
            tl_debug(UCC_TASK_LIB(task),
                     "sharp_coll_do_bcast_nb not supported, msgsize %zd",
                     data_size);
        } else {
            tl_error(UCC_TASK_LIB(task), "sharp_coll_do_bcast_nb failed:%s",
                     sharp_coll_strerror(ret));
        }
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_allreduce_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
         ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] == SHARP_DTYPE_NULL ||
        ucc_to_sharp_reduce_op[args->op] == SHARP_OP_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_allreduce_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_bcast_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args  = &TASK_ARGS(task);
    size_t           data_size;

    data_size = ucc_dt_size(args->src.info.datatype) * args->src.info.count;

    /* check SHARP supports memory type, dataype is contig and
       data size is even in case of older sharp versions */
    if ((ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        !ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID) ||
        ((data_size % 2 != 0) &&
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] == SHARP_DTYPE_NULL)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_bcast_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_barrier_init(ucc_tl_sharp_task_t *task)
{
    task->super.post     = ucc_tl_sharp_barrier_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
};

ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_coll_task_t coll_task = task->super;
    size_t           data_size;
    int              size = (int)(coll_task.bargs.team->size);

    data_size = ucc_dt_size(args->src.info.datatype) * args->src.info.count;

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
         ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] == SHARP_DTYPE_NULL ||
        ucc_to_sharp_reduce_op[args->op] == SHARP_OP_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    //choose real function by msg_size
    if(data_size/size < 16*1024){
        //message size smaller than 16k, use allreduce
        task->super.post     = ucc_tl_sharp_reduce_scatter_arw_start;
        task->super.progress = ucc_tl_sharp_collective_progress;
    }else{
        //message size bigger or equal to 16k, use reduce_nonblocking
        //reduce_nonblocking requirs dealing multiple reques handles, use a different progress fnx
        task->super.post     = ucc_tl_sharp_reduce_scatter_nr_start;
        task->super.progress = ucc_tl_sharp_reduce_scatter_nr_progress;
    }

    return UCC_OK;

}