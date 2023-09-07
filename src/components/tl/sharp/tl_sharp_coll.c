/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "tl_sharp.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <stdlib.h>

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
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
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

/* just return UCC_OK because we use bolcked allreduce to implement reduce scatter*/
void ucc_tl_sharp_collective_progress_for_rs(ucc_coll_task_t *coll_task)
{
    // TODO(lqb) 处理 allreduce_nb 的数据筛选以及 reduce_nb 的资源释放
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    if (task->reduce_scatter.req_handle_num > 1) {
        for (int i = 0; i < task->reduce_scatter.req_handle_num; i++) {
            if (task->reduce_scatter.finished_tasks[i]) continue;
            if (sharp_coll_req_test(task->reduce_scatter.req_handles[i])) {
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->reduce_scatter.s_mem_hs[i]);
                }
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->reduce_scatter.r_mem_hs[i]);
                sharp_coll_req_free(task->reduce_scatter.req_handles[i]);
                task->reduce_scatter.finished_tasks[i] = 1;
                task->reduce_scatter.finished_task_num ++;
            }
        }
        if (task->reduce_scatter.finished_task_num == task->reduce_scatter.req_handle_num) {
            free((void *)task->reduce_scatter.s_mem_hs);
            free((void *)task->reduce_scatter.r_mem_hs);
            free((void *)task->reduce_scatter.req_handles);
            free((void*)task->reduce_scatter.finished_tasks);
            coll_task->status = UCC_OK;
            UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                            "sharp_collective_done", 0);
        }
    } else {
        if (task->req_handle != NULL) {
            completed = sharp_coll_req_test(task->req_handle);
            if (completed) {
                // choose data
                memcpy(task->reduce_scatter.recv_buf, 
                    (void *)(task->reduce_scatter.tmp_buf + task->reduce_scatter.recv_data_start), 
                    task->reduce_scatter.recv_data_size);
                // release resources
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                task->reduce_scatter.s_mem_hs[0]);
                }
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->reduce_scatter.r_mem_hs[0]);
                sharp_coll_req_free(task->req_handle);
                free((void *)task->reduce_scatter.s_mem_hs);
                free((void *)task->reduce_scatter.r_mem_hs);
                free((void *)task->reduce_scatter.tmp_buf);
                coll_task->status = UCC_OK;
                UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                                "sharp_collective_done", 0);
            }
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

void fill_reduce_spec(struct sharp_coll_reduce_spec *reduce_spec, ucc_coll_args_t *args,
                    void *sbuf, void *rbuf, 
                    ucc_tl_sharp_reg_t *s_mem_h, ucc_tl_sharp_reg_t *r_mem_h,
                    size_t send_data_size, size_t send_count, 
                    enum sharp_datatype sharp_type, 
                    enum sharp_reduce_op op_type, 
                    enum sharp_data_memory_type src_mem_type,
                    enum sharp_data_memory_type dest_mem_type) {

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec->sbuf_desc.buffer.ptr        = sbuf;
        reduce_spec->sbuf_desc.buffer.mem_handle = s_mem_h->mr;
        reduce_spec->sbuf_desc.mem_type          = src_mem_type;
    } else {
        reduce_spec->sbuf_desc.buffer.ptr        = rbuf;
        reduce_spec->sbuf_desc.buffer.mem_handle = r_mem_h->mr;
        reduce_spec->sbuf_desc.mem_type          = dest_mem_type;
    }

    reduce_spec->sbuf_desc.buffer.length     = send_data_size;
    reduce_spec->sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec->rbuf_desc.buffer.ptr        = rbuf;
    reduce_spec->rbuf_desc.buffer.length     = send_data_size;
    reduce_spec->rbuf_desc.buffer.mem_handle = r_mem_h->mr;
    reduce_spec->rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec->rbuf_desc.mem_type          = dest_mem_type;
    reduce_spec->aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec->length                      = send_count;
    reduce_spec->dtype                       = sharp_type;
    reduce_spec->op                          = op_type;
}

void wait_for_complete(ucc_tl_sharp_task_t *task, void **req_handles, 
            ucc_tl_sharp_reg_t **s_mem_h, ucc_tl_sharp_reg_t **r_mem_h, int count) {
    int8_t finished[count];
    memset(&finished[0], 0, count);
    int _count = count;
    while (_count > 0) {
        for (int i = 0; i < count; i++) {
            if (finished[i]) continue;
            if (sharp_coll_req_test(req_handles[i])) {
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                s_mem_h[i]);
                }
                ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            r_mem_h[i]);
                sharp_coll_req_free(req_handles[i]);
                finished[i] = 1;
                _count--;
            }
        }
    }
}

#define CHECK_MALLOC(var, malloc, type)                                     \
    void *addr = (malloc);                                                  \
    if (addr == NULL) {                                                     \
        tl_error(UCC_TASK_LIB(task), "failed to allocate enough memory!");  \
        exit(-1);                                                           \
    }                                                                       \
    (var) = (type)addr;

/**
 * filt data after allreduce to implement reduce scatter
 */
ucc_status_t ucc_tl_sharp_reduce_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t          *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team  = TASK_TEAM(task);
    ucc_coll_args_t              *args  = &TASK_ARGS(task);
    size_t                        send_count = args->src.info.count;
    ucc_datatype_t                dt = args->dst.info.datatype;
    enum sharp_data_memory_type   src_mem_type = ucc_to_sharp_memtype[args->src.info.mem_type];
    enum sharp_data_memory_type   dest_mem_type = ucc_to_sharp_memtype[args->dst.info.mem_type];
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        send_data_size;
    int                           ret;
    int                           local_rank = coll_task->team->params.rank;
    int                           global_ranks = coll_task->team->params.size;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_reduce_scatter_start", 0);
    
    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];
    send_data_size  = ucc_dt_size(dt) * send_count;
    size_t count_per_node = send_count / global_ranks;
    size_t count_per_node_mod = send_count % global_ranks;
    int8_t should_use_reduce = (count_per_node * ucc_dt_size(dt) >= 16384);

    if (should_use_reduce) {
        // void *req_handle_buffer[global_ranks];
        // ucc_tl_sharp_reg_t *s_mem_h[global_ranks];
        // ucc_tl_sharp_reg_t *r_mem_h[global_ranks];

        // task->reduce_scatter.finished_tasks = (int8_t *)calloc(global_ranks, sizeof(int8_t));
        // task->reduce_scatter.req_handles = (void **)malloc(sizeof(void *) * global_ranks);
        // task->reduce_scatter.s_mem_hs = (ucc_tl_sharp_reg_t **)malloc(sizeof(ucc_tl_sharp_reg_t *) * global_ranks);
        // task->reduce_scatter.r_mem_hs = (ucc_tl_sharp_reg_t **)malloc(sizeof(ucc_tl_sharp_reg_t *) * global_ranks);

        task->req_handle_num = global_ranks;
        task->reduce_scatter.finished_task_num = 0;
        CHECK_MALLOC(task->reduce_scatter.finished_tasks, calloc(global_ranks, sizeof(int8_t)), int8_t*)
        CHECK_MALLOC(task->reduce_scatter.req_handles, malloc(sizeof(void *) * global_ranks), void**)
        CHECK_MALLOC(task->reduce_scatter.s_mem_hs, malloc(sizeof(ucc_tl_sharp_reg_t *) * global_ranks), ucc_tl_sharp_reg_t**)
        CHECK_MALLOC(task->reduce_scatter.r_mem_hs, malloc(sizeof(ucc_tl_sharp_reg_t *) * global_ranks), ucc_tl_sharp_reg_t**)
        struct sharp_coll_reduce_spec reduce_spec[global_ranks];

        int send_nb_count = count_per_node;
        int size_per_node = send_nb_count * ucc_dt_size(dt);
        int send_start = 0;
        for (int i = 0; i < global_ranks; i++) {
            if (i == global_ranks - 1) {
                send_nb_count += count_per_node_mod;
                size_per_node += count_per_node_mod * ucc_dt_size(dt);
            }

            void *sbuf_start = (void*)(((int8_t*)args->src.info.buffer) + send_start);
            if (!UCC_IS_INPLACE(*args)) {
                ucc_tl_sharp_mem_register(TASK_CTX(task), team, sbuf_start, size_per_node,
                                        &task->reduce_scatter.s_mem_hs[i]);
            }
            ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer,
                                    size_per_node,
                                    &task->reduce_scatter.r_mem_hs[i]);

            reduce_spec[i].root = i;
            fill_reduce_spec(&reduce_spec[i], args, 
                            sbuf_start, (void*)args->dst.info.buffer, 
                            task->reduce_scatter.s_mem_hs[i], 
                            task->reduce_scatter.r_mem_hs[i], 
                            size_per_node, send_nb_count, sharp_type, op_type, 
                            src_mem_type, dest_mem_type);

            ret = sharp_coll_do_reduce_nb(team->sharp_comm, &reduce_spec[i],
                                        &task->reduce_scatter.req_handles[i]);
            if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
                tl_error(UCC_TASK_LIB(task), "sharp_coll_do_reduce_nb failed:%s",
                        sharp_coll_strerror(ret));
                coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
                return ucc_task_complete(coll_task);
            }
            send_start += size_per_node;
        }
        assert(send_start == send_data_size);
        // TODO(lqb) 将 wait_for_complete 移动到 progress 函数中
        // wait_for_complete(task, req_handle_buffer, s_mem_h, r_mem_h, global_ranks);
    } else {
        // int8_t recv_buf[send_data_size];
        // task->reduce_scatter.s_mem_hs = (ucc_tl_sharp_reg_t **)malloc(sizeof(ucc_tl_sharp_reg_t *));
        // task->reduce_scatter.r_mem_hs = (ucc_tl_sharp_reg_t **)malloc(sizeof(ucc_tl_sharp_reg_t *));
        // task->reduce_scatter.tmp_buf = (int8_t *)malloc(send_data_size);

        task->req_handle_num = 1;
        CHECK_MALLOC(task->reduce_scatter.s_mem_hs, malloc(sizeof(ucc_tl_sharp_reg_t *)), ucc_tl_sharp_reg_t**)
        CHECK_MALLOC(task->reduce_scatter.r_mem_hs, malloc(sizeof(ucc_tl_sharp_reg_t *)), ucc_tl_sharp_reg_t**)
        CHECK_MALLOC(task->reduce_scatter.tmp_buf, malloc(send_data_size), int8_t*)
        task->reduce_scatter.recv_buf = args->dst.info.buffer;

        if (!UCC_IS_INPLACE(*args)) {
            ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer, send_data_size,
                                    &task->reduce_scatter.s_mem_hs[0]);
        }
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, (void*)task->reduce_scatter.tmp_buf, 
                                send_data_size,
                                &task->reduce_scatter.r_mem_hs[0]);
        struct sharp_coll_reduce_spec reduce_spec;
        fill_reduce_spec(&reduce_spec, args, 
                        args->src.info.buffer, (void*)task->reduce_scatter.tmp_buf, 
                        task->reduce_scatter.s_mem_hs[0],
                        task->reduce_scatter.r_mem_hs[0], 
                        send_data_size, send_count, sharp_type, op_type, 
                        src_mem_type, dest_mem_type);
        
        ret = sharp_coll_do_allreduce_nb(team->sharp_comm, &reduce_spec, &task->req_handle);
        if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
            tl_error(UCC_TASK_LIB(task), "sharp_coll_do_allreduce failed:%s",
                    sharp_coll_strerror(ret));
            coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
            return ucc_task_complete(coll_task);
        }

        int copy_size = count_per_node * ucc_dt_size(dt);
        int copy_start = copy_size * local_rank;
        if (local_rank == global_ranks - 1) {
            copy_size += count_per_node_mod * ucc_dt_size(dt);
        }
        task->reduce_scatter.recv_data_size = copy_size;
        task->reduce_scatter.recv_data_start = copy_start;
        // memcpy(args->dst.info.buffer, recv_buf + copy_start, copy_size);

        // if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        //     ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
        //                                 task->allreduce.s_mem_h);
        // }
        // ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
        //                             task->allreduce.r_mem_h);
    }

    coll_task->status = UCC_INPROGRESS;
    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                    "sharp_reduce_scatter_start done", 0);

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

ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task)
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

    task->super.post     = ucc_tl_sharp_reduce_scatter_start;
    task->super.progress = ucc_tl_sharp_collective_progress_for_rs;
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
