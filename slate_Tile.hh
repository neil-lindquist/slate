//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate_Memory.hh"

#include <blas.hh>
#include <lapack.hh>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <memory>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
template <typename scalar_t>
class Tile {
public:
    Tile() {}

    Tile(int64_t mb, int64_t nb,
         std::weak_ptr<Memory> memory,
         MPI_Comm mpi_comm);

    Tile(int64_t mb, int64_t nb,
         scalar_t *a, int64_t lda,
         std::weak_ptr<Memory> memory,
         MPI_Comm mpi_comm);

    Tile(const Tile<scalar_t> *src_tile, int dst_device_num);

    ~Tile() { deallocate(); }

    Tile<scalar_t>* copyToHost(cudaStream_t stream);
    Tile<scalar_t>* copyToDevice(int device_num, cudaStream_t stream);

    void copyDataToHost(const Tile<scalar_t> *dst_tile, cudaStream_t stream);
    void copyDataToDevice(const Tile<scalar_t> *dst_tile, cudaStream_t stream);

    void send(int dst);
    void recv(int src);
    void bcast(int bcast_root, MPI_Comm bcast_comm);

    static void gemm(blas::Op transa, blas::Op transb,
                     scalar_t alpha, Tile<scalar_t> *a,
                                      Tile<scalar_t> *b,
                     scalar_t beta,  Tile<scalar_t> *c);

    static void potrf(blas::Uplo uplo, Tile<scalar_t> *a);

    static void syrk(blas::Uplo uplo, blas::Op trans,
                     scalar_t alpha, Tile<scalar_t> *a,
                     scalar_t beta,  Tile<scalar_t> *c);

    static void trsm(blas::Side side, blas::Uplo uplo,
                     blas::Op transa, blas::Diag diag,
                     scalar_t alpha, Tile<scalar_t> *a,
                                      Tile<scalar_t> *b);
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;

    scalar_t *data_;

    bool valid_;
    bool origin_;

protected:
    size_t size() { return sizeof(scalar_t)*mb_*nb_; }
    void allocate();
    void deallocate();

    static int host_num_;
    int device_num_;

    MPI_Comm mpi_comm_;
    std::weak_ptr<Memory> memory_;
};

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t mb, int64_t nb,
                      std::weak_ptr<Memory> memory,
                      MPI_Comm mpi_comm)

    : mb_(mb),
      nb_(nb),
      stride_(mb),
      data_(nullptr),
      valid_(true),
      origin_(false),
      device_num_(host_num_),
      mpi_comm_(mpi_comm),
      memory_(memory)
{
    allocate();
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t mb, int64_t nb,
                      scalar_t *a, int64_t lda,
                      std::weak_ptr<Memory> memory,
                      MPI_Comm mpi_comm)

    : mb_(mb),
      nb_(nb),
      stride_(lda),
      data_(a),
      valid_(true),
      origin_(true),
      device_num_(host_num_),
      mpi_comm_(mpi_comm),
      memory_(memory)
{}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
Tile<scalar_t>::Tile(const Tile<scalar_t> *src_tile, int dst_device_num)
{
    *this = *src_tile;
    this->origin_ = false;
    this->stride_ = this->mb_;
    this->device_num_ = dst_device_num;
    allocate();
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
Tile<scalar_t>*
Tile<scalar_t>::copyToHost(cudaStream_t stream)
{
    Tile<scalar_t> *dst_tile = new Tile<scalar_t>(this, this->host_num_);
    this->copyDataToHost(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
Tile<scalar_t>*
Tile<scalar_t>::copyToDevice(int device_num, cudaStream_t stream)
{
    Tile<scalar_t> *dst_tile = new Tile<scalar_t>(this, device_num);
    this->copyDataToDevice(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::copyDataToHost(
    const Tile<scalar_t> *dst_tile, cudaStream_t stream)
{
    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, size(),
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("Gray");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::copyDataToDevice(
    const Tile<scalar_t> *dst_tile, cudaStream_t stream)
{
    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(dst_tile->device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, size(),
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(scalar_t)*dst_tile->stride_;
        size_t spitch = sizeof(scalar_t)*stride_;
        size_t width = sizeof(scalar_t)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("LightGray");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::send(int dst)
{
    // If no stride.
    if (stride_ == mb_) {

        // Use simple send.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Send(data_, count, MPI_DOUBLE, dst, tag, mpi_comm_);
        assert(retval == MPI_SUCCESS);
    }
    else {

        // Otherwise, use strided send.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Send(data_, 1, newtype, dst, tag, mpi_comm_);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::recv(int src)
{
    // If no stride.
    if (stride_ == mb_) {

        // Use simple recv.
        int count = mb_*nb_;
        int tag = 0;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Recv(
            data_, count, MPI_DOUBLE, src, tag, mpi_comm_, MPI_STATUS_IGNORE);
        assert(retval == MPI_SUCCESS);
    }
    else {

        // Otherwise, use strided recv. 
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        int tag = 0;
        #pragma omp critical(slate_mpi)
        retval = MPI_Recv(
            data_, 1, newtype, src, tag, mpi_comm_, MPI_STATUS_IGNORE);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::bcast(int bcast_root, MPI_Comm bcast_comm)
{
    // If no stride.
    if (stride_ == mb_) {

        // Use simple bcast.
        int count = mb_*nb_;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Bcast(data_, count, MPI_DOUBLE, bcast_root, bcast_comm);
        assert(retval == MPI_SUCCESS);
    }
    else {

        // Otherwise, use strided bcast.
        int count = nb_;
        int blocklength = mb_;
        int stride = stride_;
        MPI_Datatype newtype;
        int retval;

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_vector(
            count, blocklength, stride, MPI_DOUBLE, &newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_commit(&newtype);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Bcast(data_, 1, newtype, bcast_root, bcast_comm);
        assert(retval == MPI_SUCCESS);

        #pragma omp critical(slate_mpi)
        retval = MPI_Type_free(&newtype);
        assert(retval == MPI_SUCCESS);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::gemm(blas::Op transa, blas::Op transb,
                 scalar_t alpha, Tile<scalar_t> *a,
                                  Tile<scalar_t> *b,
                 scalar_t beta,  Tile<scalar_t> *c)
{
    trace_cpu_start();
    blas::gemm(blas::Layout::ColMajor,
               transa, transb,
               c->mb_, c->nb_, a->nb_,
               alpha, a->data_, a->stride_,
                      b->data_, b->stride_,
               beta,  c->data_, c->stride_);
    trace_cpu_stop("MediumAquamarine");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::potrf(lapack::Uplo uplo, Tile<scalar_t> *a)
{
    trace_cpu_start();
    lapack::potrf(uplo,
                  a->nb_,
                  a->data_, a->stride_);
    trace_cpu_stop("RosyBrown");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::syrk(blas::Uplo uplo, blas::Op trans,
                 scalar_t alpha, Tile<scalar_t> *a,
                 scalar_t beta,  Tile<scalar_t> *c)
{
    trace_cpu_start();
    blas::syrk(blas::Layout::ColMajor,
               uplo, trans,
               c->nb_, a->nb_,
               alpha, a->data_, a->stride_,
               beta,  c->data_, c->stride_);
    trace_cpu_stop("CornflowerBlue");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::trsm(blas::Side side, blas::Uplo uplo,
                 blas::Op transa, blas::Diag diag,
                 scalar_t alpha, Tile<scalar_t> *a,
                                  Tile<scalar_t> *b)
{
    trace_cpu_start();
    blas::trsm(blas::Layout::ColMajor,
               side, uplo, transa, diag,
               b->mb_, b->nb_,
               alpha, a->data_, a->stride_,
                      b->data_, b->stride_);
    trace_cpu_stop("MediumPurple");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::allocate()
{
    trace_cpu_start();
    data_ = (scalar_t*)memory_.lock()->alloc(device_num_);
    trace_cpu_stop("Orchid");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Tile<scalar_t>::deallocate()
{
    trace_cpu_start();
    memory_.lock()->free(data_, device_num_);
    data_ = nullptr;
    trace_cpu_stop("Crimson");
}

} // namespace slate

#endif // SLATE_TILE_HH
