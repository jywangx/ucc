/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_MPI_H
#define TEST_MPI_H
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>
#include <mpi.h>
#include <ucc/api/ucc.h>
BEGIN_C_DECLS
#include "core/ucc_mc.h"
#include "utils/ucc_math.h"
END_C_DECLS

#define STR(x) #x
#define UCC_CHECK(_call)                                            \
    if (UCC_OK != (_call)) {                                        \
        std::cerr << "*** UCC TEST FAIL: " << STR(_call) << "\n";   \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }

typedef enum {
    TEAM_WORLD,
    TEAM_REVERSE,
    TEAM_SPLIT_HALF,
    TEAM_SPLIT_ODD_EVEN,
    TEAM_LAST
} ucc_test_mpi_team_t;

static inline const char* team_str(ucc_test_mpi_team_t t) {
    switch(t) {
    case TEAM_WORLD:
        return "world";
    case TEAM_REVERSE:
        return "reverse";
    case TEAM_SPLIT_HALF:
        return "half";
    case TEAM_SPLIT_ODD_EVEN:
        return "odd_even";
    default:
        break;
    }
    return NULL;
}

typedef struct ucc_test_team {
    ucc_test_mpi_team_t type;
    MPI_Comm comm;
    ucc_team_h team;
    ucc_context_h ctx;
    ucc_test_team(ucc_test_mpi_team_t _type, MPI_Comm _comm,
                  ucc_team_h _team, ucc_context_h _ctx) :
    type(_type), comm(_comm), team(_team), ctx(_ctx) {};
} ucc_test_team_t;

class UccTestMpi {
    ucc_context_h ctx;
    ucc_lib_h     lib;
    void create_team(ucc_test_mpi_team_t t);
    void destroy_team(ucc_test_team_t &team);
    ucc_team_h create_ucc_team(MPI_Comm comm);    
    std::vector<ucc_test_team_t> teams;
    std::vector<size_t> msgsizes;
    std::vector<ucc_memory_type_t> mtypes;
    std::vector<ucc_datatype_t> dtypes;
    std::vector<ucc_reduction_op_t> ops;
    std::vector<ucc_coll_type_t> colls;
public:
    UccTestMpi(int argc, char *argv[], ucc_thread_mode_t tm,
               std::vector<ucc_test_mpi_team_t> &test_teams,
               const char* cls = NULL);
    ~UccTestMpi();
    void set_msgsizes(size_t min, size_t max, size_t power);
    void set_dtypes(std::vector<ucc_datatype_t> &_dtypes);
    void set_colls(std::vector<ucc_coll_type_t> &_colls);
    void set_ops(std::vector<ucc_reduction_op_t> &_ops);
    void set_mtypes(std::vector<ucc_memory_type_t> &_mtypes);
    ucc_status_t run_all();
};

class TestCase {
protected:
    ucc_test_team_t team;    
    ucc_memory_type_t mem_type;
    size_t msgsize;
    ucc_coll_args_t args;
    ucc_coll_req_h req;
    void *sbuf;
    void *rbuf;
    void *check_buf;
public:
    static std::shared_ptr<TestCase> init(ucc_coll_type_t _type,
                                          ucc_test_team_t &_team,
                                          size_t msgsize = 0,
                                          ucc_memory_type_t mt = UCC_MEMORY_TYPE_HOST,
                                          ucc_datatype_t dt = UCC_DT_INT32,
                                          ucc_reduction_op_t op = UCC_OP_SUM);

    TestCase(ucc_test_team_t &_team, ucc_memory_type_t _mem_type, size_t _msgsize);
    ~TestCase();
    virtual void run();
    virtual ucc_status_t check() = 0;
    virtual std::string str();
    virtual ucc_status_t test();
    void wait();
    ucc_status_t exec();
};

class TestBarrier : public TestCase {
    ucc_status_t status;
public:
    TestBarrier(ucc_test_team_t &team);
    ucc_status_t check();
    std::string str();
    void run();
    ucc_status_t test();
};

class TestAllreduce : public TestCase {
    ucc_datatype_t dt;
    ucc_reduction_op_t op;
public:
    TestAllreduce(size_t _msgsize, ucc_datatype_t _dt, ucc_reduction_op_t _op,
                  ucc_memory_type_t _mt, ucc_test_team_t &team);
    ucc_status_t check();
    std::string str();
};

void init_buffer(void *buf, size_t count, ucc_datatype_t dt,
                 ucc_memory_type_t mt, int value);

ucc_status_t compare_buffers(void *rst, void *expected, size_t count,
                             ucc_datatype_t dt, ucc_memory_type_t mt);
#endif
