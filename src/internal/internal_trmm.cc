// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Dispatches to target implementations.
/// @ingroup trmm_internal
///
template <Target target, typename scalar_t>
void trmm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority, int64_t queue_index,
          Options const& opts )
{
    trmm(internal::TargetType<target>(),
         side,
         alpha, A,
                B,
         priority, queue_index, opts);
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host OpenMP task implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostTask>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, int64_t queue_index,
          Options const& opts )
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::trmm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    assert(A.mt() == 1);

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    // alternatively, if (side == right), (conj)-transpose both A and B,
    // then assume side == left; see slate::trmm
    #pragma omp taskgroup
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) \
                    firstprivate( i, layout, side, alpha, call_tile_tick )
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    tile::trmm(
                        side, A.diag(),
                        alpha, A(0, 0), B(i, 0) );
                    if (call_tile_tick) {
                        // todo: should tileRelease()?
                        A.tileTick(0, 0);
                    }
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) \
                    firstprivate( j, layout, side, alpha, call_tile_tick )
                {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    tile::trmm(
                        side, A.diag(),
                        alpha, A(0, 0), B(0, j) );
                    if (call_tile_tick) {
                        // todo: should tileRelease()?
                        A.tileTick(0, 0);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host nested OpenMP implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostNest>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, int64_t queue_index,
          Options const& opts )
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host batched implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::HostBatch>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, int64_t queue_index,
          Options const& opts )
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}
//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// GPU device batched cuBLAS implementation.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(internal::TargetType<Target::Devices>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, int64_t queue_index,
          Options const& opts )
{
    // CPU assumes column major
    const Layout layout = Layout::ColMajor;

    using std::swap;
    using blas::conj;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    assert(B.num_devices() > 0);
    assert(A.mt() == 1);
    assert(B.uploPhysical() == Uplo::General);
    assert(A.mt() == A.nt());  // square
    assert(side == Side::Left ? A.mt() == B.mt() : A.mt() == B.nt());

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    Uplo uploA = A.uploPhysical();
    Diag diagA = A.diag();
    Op opA = A.op();
    Side sideA = side;

    if (B.op() != Op::NoTrans) {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        sideA = (side == Side::Left ? Side::Right : Side::Left);
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);
    }

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority) \
            firstprivate( device, side, sideA, uploA, opA, diagA, alpha ) \
            firstprivate( queue_index, layout, call_tile_tick )
        {
            std::set<ij_tuple> B_tiles_set;
            if (side == Side::Right) {
                for (int64_t i = 0; i < B.mt(); ++i) {
                    if (B.tileIsLocal(i, 0)
                        && device == B.tileDevice(i, 0))
                    {
                        B_tiles_set.insert({i, 0});
                    }
                }
            }
            else {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(0, j)
                        && device == B.tileDevice(0, j))
                    {
                        B_tiles_set.insert({0, j});
                    }
                }
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {

                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));

                scalar_t** a_array_host = B.array_host(device, queue_index);
                scalar_t** b_array_host = a_array_host + batch_size;

                // B comes first since we do computation for a local B
                auto group_params = device_regions_build<false, 2, scalar_t>(
                        {B, A},
                        {b_array_host, a_array_host},
                        device );

                {
                    trace::Block trace_block("blas::batch::trmm");

                    std::vector<Side>      side_(1, sideA);
                    std::vector<Uplo>      uplo_(1, uploA);
                    std::vector<Op>         opA_(1, opA  );
                    std::vector<Diag>      diag_(1, diagA);
                    std::vector<scalar_t> alpha_(1, alpha);
                    // info size 0 disables slow checks in batched BLAS++.
                    std::vector<int64_t> info;

                    blas::Queue* queue = B.compute_queue(device, queue_index);
                    assert(queue != nullptr);

                    for (size_t g = 0; g < group_params.size(); ++g) {

                        int64_t group_count = group_params[ g ].count;

                        std::vector<int64_t>    m(1, group_params[ g ].mb);
                        std::vector<int64_t>    n(1, group_params[ g ].nb);
                        std::vector<int64_t> ldda(1, group_params[ g ].ld[1]);
                        std::vector<int64_t> lddb(1, group_params[ g ].ld[0]);

                        std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                        std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);

                        if (B.op() != Op::NoTrans) {
                            swap(m, n);
                        }

                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            alpha_, a_array, ldda,
                                    b_array, lddb,
                            group_count, info, *queue);

                        a_array_host += group_count;
                        b_array_host += group_count;
                    }

                    queue->sync();
                }

                if (call_tile_tick) {
                    A.tileRelease(0, 0, device);
                    for (auto i = 0; i < batch_size; ++i) {
                        A.tileTick(0, 0);
                    }
                }
            }
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trmm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostBatch, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::HostBatch, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::Devices, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void trmm< Target::Devices, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, int64_t queue_index,
    Options const& opts );


} // namespace internal
} // namespace slate
