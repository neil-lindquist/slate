// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

namespace slate {

namespace device {

template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<float> alpha, std::complex<float>** Aarray, int64_t lda,
    std::complex<float> beta, std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    geadd(m, n,
          make_cuFloatComplex(alpha.real(), alpha.imag()), (cuFloatComplex**) Aarray, lda,
          make_cuFloatComplex(beta.real(), beta.imag()), (cuFloatComplex**) Barray, ldb,
          batch_count, queue);
#endif
}

template <>
void geadd(
    int64_t m, int64_t n,
    std::complex<double> alpha, std::complex<double>** Aarray, int64_t lda,
    std::complex<double> beta, std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
#if !defined(SLATE_NO_CUDA)
    geadd(m, n,
          make_cuDoubleComplex(alpha.real(), alpha.imag()) , (cuDoubleComplex**) Aarray, lda,
          make_cuDoubleComplex(beta.real(), beta.imag()), (cuDoubleComplex**) Barray, ldb,
          batch_count, queue);
#endif
}

#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void geadd(
    int64_t m, int64_t n,
    double alpha, double** Aarray, int64_t lda,
    double beta, double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}

template <>
void geadd(
    int64_t m, int64_t n,
    float alpha, float** Aarray, int64_t lda,
    float beta, float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue &queue)
{
}
#endif // not SLATE_WITH_CUDA

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// General matrix add.
/// Dispatches to target implementations.
/// @ingroup geadd_internal
///
/// todo: this function should just be named "add".
template <Target target, typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>&& A,
           scalar_t beta, Matrix<scalar_t>&& B,
           int priority, int queue_index)
{
    geadd(internal::TargetType<target>(),
          alpha, A,
          beta,  B,
          priority, queue_index);
}

//------------------------------------------------------------------------------
/// General matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup geadd_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void geadd(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta, Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    // trace::Block trace_block("geadd");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task shared(A, B) priority(priority)
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    B.tileGetForWriting(i, j, LayoutConvert::None);
                    axpby(alpha, A(i, j),
                         beta,  B(i, j));
                    A.tileTick(i, j);// TODO is this correct here?
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void geadd(internal::TargetType<Target::HostNest>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta, Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// todo: this function should just be named "add".
template <typename scalar_t>
void geadd(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta, Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}

//------------------------------------------------------------------------------
/// General matrix add.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup geadd_internal
///
/// todo: this function should just be named "add".
template <typename scalar_t>
void geadd(internal::TargetType<Target::Devices>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta, Matrix<scalar_t>& B,
           int priority, int queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    int64_t irange[4][2] = {
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   },
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        B.nt()-1 },
        { 0,        B.nt()-1 },
        { B.nt()-1, B.nt()   },
        { B.nt()-1, B.nt()   }
    };

    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority)
        {
            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = Layout::ColMajor;
            std::set<ij_tuple> A_tiles_set, B_tiles_set;

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                        B_tiles_set.insert({i, j});
                    }
                }
            }
            #pragma omp task default(shared)
            {
                A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
            }
            #pragma omp task default(shared)
            {
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));
            }
            #pragma omp taskwait

            int64_t batch_size = A_tiles_set.size();
            scalar_t** a_array_host = B.array_host(device);
            scalar_t** b_array_host = a_array_host + batch_size;

            int64_t batch_count = 0;
            int64_t mb[4], nb[4], lda[4], ldb[4], group_count[4];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, j, device).data();
                            b_array_host[batch_count] = B(i, j, device).data();
                            lda[q] = A(i, j, device).stride();
                            ldb[q] = B(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                }
            }
            slate_assert(batch_count == batch_size);

            scalar_t** a_array_dev = B.array_device(device);
            scalar_t** b_array_dev = a_array_dev + batch_size;

            blas::Queue* queue = A.queue(device, queue_index);

            blas::device_memcpy<scalar_t*>(a_array_dev, a_array_host,
                                batch_count*2,
                                blas::MemcpyKind::HostToDevice,
                                *queue);

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::geadd(mb[q], nb[q],
                                  alpha, a_array_dev, lda[q],
                                  beta,  b_array_dev, ldb[q],
                                  group_count[q], *queue);
                    a_array_dev += group_count[q];
                    b_array_dev += group_count[q];
                }
            }

            queue->sync();

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        // erase tmp local and remote device tiles;
                        A.tileRelease(i, j, device);
                        // decrement life for remote tiles
                        A.tileTick(i, j);
                    }
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
/// todo: these functions should just be named "add".
// ----------------------------------------
template
void geadd<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
    float beta, Matrix<float>&& B,
    int priority, int queue_index);

template
void geadd<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
    float beta, Matrix<float>&& B,
    int priority, int queue_index);

template
void geadd<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
    float beta, Matrix<float>&& B,
    int priority, int queue_index);

template
void geadd<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
    float beta, Matrix<float>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void geadd<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta, Matrix<double>&& B,
    int priority, int queue_index);

template
void geadd<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
    double beta, Matrix<double>&& B,
    int priority, int queue_index);

template
void geadd<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
    double beta, Matrix<double>&& B,
    int priority, int queue_index);

template
void geadd<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
    double beta, Matrix<double>&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void geadd< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void geadd< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void geadd< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

template
void geadd< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float>  beta, Matrix< std::complex<float> >&& B,
    int priority, int queue_index);

// ----------------------------------------
template
void geadd< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void geadd< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void geadd< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

template
void geadd< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta, Matrix< std::complex<double> >&& B,
    int priority, int queue_index);

} // namespace internal
} // namespace slate
