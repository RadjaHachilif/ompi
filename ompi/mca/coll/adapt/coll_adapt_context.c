#include "ompi/mca/coll/coll.h"
#include "coll_adapt_context.h"

static void mca_coll_adapt_bcast_context_constructor(mca_coll_adapt_bcast_context_t *bcast_context){
}

static void mca_coll_adapt_bcast_context_destructor(mca_coll_adapt_bcast_context_t *bcast_context){
    
}

static void mca_coll_adapt_constant_bcast_context_constructor(mca_coll_adapt_constant_bcast_context_t *con){
}

static void mca_coll_adapt_constant_bcast_context_destructor(mca_coll_adapt_constant_bcast_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_bcast_context_t, opal_free_list_item_t, mca_coll_adapt_bcast_context_constructor, mca_coll_adapt_bcast_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_bcast_context_t, opal_object_t, mca_coll_adapt_constant_bcast_context_constructor, mca_coll_adapt_constant_bcast_context_destructor);

static void mca_coll_adapt_reduce_context_constructor(mca_coll_adapt_reduce_context_t *reduce_context){
}

static void mca_coll_adapt_reduce_context_destructor(mca_coll_adapt_reduce_context_t *reduce_context){
    
}

static void mca_coll_adapt_constant_reduce_context_constructor(mca_coll_adapt_constant_reduce_context_t *con){
}

static void mca_coll_adapt_constant_reduce_context_destructor(mca_coll_adapt_constant_reduce_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_reduce_context_t, opal_free_list_item_t, mca_coll_adapt_reduce_context_constructor, mca_coll_adapt_reduce_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_reduce_context_t, opal_object_t, mca_coll_adapt_constant_reduce_context_constructor, mca_coll_adapt_constant_reduce_context_destructor);

static void mca_coll_adapt_bcast_two_trees_context_constructor(mca_coll_adapt_bcast_two_trees_context_t *bcast_context){
}

static void mca_coll_adapt_bcast_two_trees_context_destructor(mca_coll_adapt_bcast_two_trees_context_t *bcast_context){
    
}

static void mca_coll_adapt_constant_bcast_two_trees_context_constructor(mca_coll_adapt_constant_bcast_two_trees_context_t *con){
}

static void mca_coll_adapt_constant_bcast_two_trees_context_destructor(mca_coll_adapt_constant_bcast_two_trees_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_bcast_two_trees_context_t, opal_free_list_item_t, mca_coll_adapt_bcast_two_trees_context_constructor, mca_coll_adapt_bcast_two_trees_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_bcast_two_trees_context_t, opal_object_t, mca_coll_adapt_constant_bcast_two_trees_context_constructor, mca_coll_adapt_constant_bcast_two_trees_context_destructor);


static void mca_coll_adapt_allreduce_context_constructor(mca_coll_adapt_allreduce_context_t *allreduce_context){
}

static void mca_coll_adapt_allreduce_context_destructor(mca_coll_adapt_allreduce_context_t *allreduce_context){
    
}

static void mca_coll_adapt_constant_allreduce_context_constructor(mca_coll_adapt_constant_allreduce_context_t *con){
}

static void mca_coll_adapt_constant_allreduce_context_destructor(mca_coll_adapt_constant_allreduce_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_allreduce_context_t, opal_free_list_item_t, mca_coll_adapt_allreduce_context_constructor, mca_coll_adapt_allreduce_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_allreduce_context_t, opal_object_t, mca_coll_adapt_constant_allreduce_context_constructor, mca_coll_adapt_constant_allreduce_context_destructor);

static void mca_coll_adapt_allreduce_ring_context_constructor(mca_coll_adapt_allreduce_ring_context_t *allreduce_ring_context){
}

static void mca_coll_adapt_allreduce_ring_context_destructor(mca_coll_adapt_allreduce_ring_context_t *allreduce_ring_context){
    
}

static void mca_coll_adapt_constant_allreduce_ring_context_constructor(mca_coll_adapt_constant_allreduce_ring_context_t *con){
}

static void mca_coll_adapt_constant_allreduce_ring_context_destructor(mca_coll_adapt_constant_allreduce_ring_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_allreduce_ring_context_t, opal_free_list_item_t, mca_coll_adapt_allreduce_ring_context_constructor, mca_coll_adapt_allreduce_ring_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_allreduce_ring_context_t, opal_object_t, mca_coll_adapt_constant_allreduce_ring_context_constructor, mca_coll_adapt_constant_allreduce_ring_context_destructor);


static void mca_coll_adapt_alltoallv_context_constructor(mca_coll_adapt_alltoallv_context_t *alltoallv_context){
}

static void mca_coll_adapt_alltoallv_context_destructor(mca_coll_adapt_alltoallv_context_t *alltoallv_context){
    
}

static void mca_coll_adapt_constant_alltoallv_context_constructor(mca_coll_adapt_constant_alltoallv_context_t *con){
}

static void mca_coll_adapt_constant_alltoallv_context_destructor(mca_coll_adapt_constant_alltoallv_context_t *con){
}


OBJ_CLASS_INSTANCE(mca_coll_adapt_alltoallv_context_t, opal_free_list_item_t, mca_coll_adapt_alltoallv_context_constructor, mca_coll_adapt_alltoallv_context_destructor);

OBJ_CLASS_INSTANCE(mca_coll_adapt_constant_alltoallv_context_t, opal_object_t, mca_coll_adapt_constant_alltoallv_context_constructor, mca_coll_adapt_constant_alltoallv_context_destructor);
