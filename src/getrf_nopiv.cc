// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization without pivoting.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup gesv_impl
///
template <Target target, typename scalar_t>
void getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;

    // Constants
    const scalar_t one = 1.0;
    const int life_1 = 1;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int priority_2 = 2;
    const int queue_0 = 0;
    const int queue_1 = 1;
    const int queue_2 = 2;
    const int queue_3 = 3;
    const int queue_4 = 4;
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );

    if (target == Target::Devices) {
        // two batch arrays plus one for each lookahead
        // batch array size will be set as needed
        A.allocateBatchArrays(0, 5);
        A.reserveDeviceWorkspace();
    }

    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    bool is_shared = lookahead > 0;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > A11_vector(A_nt);
    std::vector< uint8_t > A12_vector(A_nt);
    std::vector< uint8_t > A13_vector(A_nt);
    std::vector< uint8_t > A21_vector(A_nt);
    std::vector< uint8_t > A22_vector(A_nt);
    std::vector< uint8_t > A23_vector(A_nt);
    std::vector< uint8_t > A31_vector(A_nt);
    std::vector< uint8_t > A32_vector(A_nt);
    std::vector< uint8_t > A33_vector(A_nt);
    uint8_t* A11 = A11_vector.data();
    uint8_t* A12 = A12_vector.data();
    uint8_t* A13 = A13_vector.data();
    uint8_t* A21 = A21_vector.data();
    uint8_t* A22 = A22_vector.data();
    uint8_t* A23 = A23_vector.data();
    uint8_t* A31 = A31_vector.data();
    uint8_t* A32 = A32_vector.data();
    uint8_t* A33 = A33_vector.data();
    // Running two listBcastMT's simultaneously can hang due to task ordering
    // This dependency avoids that
    uint8_t listBcastMT_token;
    SLATE_UNUSED(listBcastMT_token); // Only used by OpenMP

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            // panel, high priority
            #pragma omp task depend(inout:A11[k]) \
                             priority(2)
            {
                // factor A(k, k)
                internal::getrf_nopiv<Target::HostTask>(
                    A.sub(k, k, k, k), ib, priority_2 );

                // Update panel
                int tag_k = k;
                BcastList bcast_list_A;
                bcast_list_A.push_back({k, k, {A.sub(k+1, A_mt-1, k, k),
                                               A.sub(k, k, k+1, A_nt-1)}});
                A.template listBcast<target>(
                    bcast_list_A, layout, tag_k, life_1, true );
            }

            int64_t num_la_row = std::min(A_mt-k-1, lookahead);
            int64_t num_la_col = std::min(A_nt-k-1, lookahead);
            bool la_row = num_la_row > 0;
            bool la_col = num_la_col > 0;
            bool tr_row = k+1+lookahead < A_mt;
            bool tr_col = k+1+lookahead < A_nt;

            if (la_row) {
                #pragma omp task depend(inout:A21[k]) \
                                 depend(in:A11[k]) \
                                 priority(2)
                {
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, Akk);

                    internal::trsm<target>(
                        Side::Right,
                        one, std::move( Tkk ), A.sub(k+1, k+num_la_row, k, k),
                        priority_2, layout, queue_0 );
                }

                #pragma omp task depend(inout:A21[k]) \
                                 depend(inout:listBcastMT_token) \
                                 priority(2)
                {
                    BcastListTag bcast_list;
                    // bcast the tiles of the panel to the right hand side
                    for (int64_t i = k+1; i < k+1+num_la_row; ++i) {
                        // send A(i, k) across row A(i, k+1:nt-1)
                        const int64_t tag = i;
                        bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                    }
                    A.template listBcastMT<target>(
                      bcast_list, layout, life_1, is_shared );
                }
            }
            if (tr_row) {
                #pragma omp task depend(inout:A31[k]) \
                                 depend(in:A11[k]) \
                                 depend(inout:listBcastMT_token) \
                                 priority(1)
                {
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, Akk);

                    internal::trsm<target>(
                        Side::Right,
                        one, std::move( Tkk ), A.sub(k+1+lookahead, A_mt-1, k, k),
                        priority_1, layout, queue_1 );


                    BcastListTag bcast_list;
                    // bcast the tiles of the panel to the right hand side
                    for (int64_t i = k+1+lookahead; i < A_mt; ++i) {
                        // send A(i, k) across row A(i, k+1:nt-1)
                        const int64_t tag = i;
                        bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                    }
                    A.template listBcastMT<target>(
                      bcast_list, layout, life_1, is_shared );
                }
            }
            if (la_col) {
                #pragma omp task depend(in:A11[k]) \
                                 depend(inout:A12[k]) \
                                 priority(2)
                {
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub(k, k, k+1, k+num_la_col),
                        priority_2, layout, queue_2 );

                    BcastListTag bcast_list;
                    // bcast the tiles of the panel to the right hand side
                    for (int64_t j = k+1; j < k+1+num_la_col; ++j) {
                        // send A(i, k) across row A(i, k+1:nt-1)
                        const int64_t tag = j + A_mt;
                        bcast_list.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)},
                                              tag});
                    }
                    A.template listBcastMT<target>(
                      bcast_list, layout, life_1, is_shared );
                }
            }
            if (tr_col) {
                #pragma omp task depend(in:A11[k]) \
                                 depend(inout:A13[k]) \
                                 priority(1)
                {
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub(k, k, k+1+lookahead, A_nt-1),
                        priority_1, layout, queue_3 );
                }

                #pragma omp task depend(inout:A13[k]) \
                                 depend(inout:listBcastMT_token) \
                                 priority(1)
                {
                    // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastListTag bcast_list;
                    for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        // tag must be distinct from sending left panel
                        const int64_t tag = j + A_mt;
                        bcast_list.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)},
                                              tag});
                    }
                    A.template listBcastMT<target>(
                        bcast_list, layout, life_1, is_shared);
                }
            }


            if (la_col && la_row) {
                #pragma omp task depend(in:A21[k]) \
                                 depend(in:A12[k]) \
                                 depend(in:A22[k]) \
                                 depend(out:A11[k+1]) \
                                 depend(out:A21[k+1]) \
                                 depend(out:A12[k+1]) \
                                 depend(out:A22[k+1]) \
                                 priority(2)
                {
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1, k+num_la_row, k, k),
                              A.sub(k, k, k+1, k+num_la_col),
                        one,  A.sub(k+1, k+num_la_row, k+1, k+num_la_col),
                        layout, priority_2, queue_0 );
                }
            }
            if (tr_col && la_row) {
                #pragma omp task depend(in:A21[k]) \
                                 depend(in:A13[k]) \
                                 depend(in:A23[k]) \
                                 depend(out:A12[k+1]) \
                                 depend(out:A22[k+1]) \
                                 depend(out:A13[k+1]) \
                                 depend(out:A23[k+1]) \
                                 priority(1)
                {
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1, k+num_la_row, k, k),
                              A.sub(k, k, k+1+lookahead, A_nt-1),
                        one,  A.sub(k+1, k+num_la_row, k+1+lookahead, A_nt-1),
                        layout, priority_2, queue_3 );
                }
            }
            if (la_col && tr_row) {
                #pragma omp task depend(in:A31[k]) \
                                 depend(in:A12[k]) \
                                 depend(in:A32[k]) \
                                 depend(out:A21[k+1]) \
                                 depend(out:A31[k+1]) \
                                 depend(out:A22[k+1]) \
                                 depend(out:A32[k+1]) \
                                 priority(1)
                {
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1+lookahead, A_mt-1, k, k),
                              A.sub(k, k, k+1, k+num_la_col),
                        one,  A.sub(k+1+lookahead, A_mt-1, k+1, k+num_la_col),
                        layout, priority_1, queue_1 );
                }
            }
            if (tr_col && tr_row) {
                #pragma omp task depend(in:A31[k]) \
                                 depend(in:A13[k]) \
                                 depend(in:A33[k]) \
                                 depend(out:A22[k+1]) \
                                 depend(out:A32[k+1]) \
                                 depend(out:A23[k+1]) \
                                 depend(out:A33[k+1])
                {
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1+lookahead, A_mt-1, k, k),
                              A.sub(k, k, k+1+lookahead, A_nt-1),
                        one,  A.sub(k+1+lookahead, A_mt-1, k+1+lookahead, A_nt-1),
                        layout, priority_0, queue_4 );
                }
            }
            if (target == Target::Devices) {
                #pragma omp task depend(inout:A11[k])
                {
                    if (A.tileIsLocal(k, k) && k+1 < A_nt) {
                        std::set<int> dev_set;
                        A.sub(k+1, A_mt-1, k, k).getLocalDevices(&dev_set);
                        A.sub(k, k, k+1, A_nt-1).getLocalDevices(&dev_set);

                        for (auto device : dev_set) {
                            A.tileUnsetHold(k, k, device);
                            A.tileRelease(k, k, device);
                        }
                    }
                }
                if (is_shared) {
                    #pragma omp task depend(inout:A21[k]) depend(inout:A31[k])
                    {
                        for (int64_t i = k+1; i < A_mt; ++i) {
                            if (A.tileIsLocal(i, k)) {
                                A.tileUpdateOrigin(i, k);

                                std::set<int> dev_set;
                                A.sub(i, i, k+1, A_nt-1).getLocalDevices(&dev_set);

                                for (auto device : dev_set) {
                                    A.tileUnsetHold(i, k, device);
                                    A.tileRelease(i, k, device);
                                }
                            }
                        }
                    }
                    #pragma omp task depend(inout:A12[k]) depend(inout:A13[k])
                    {
                        for (int64_t j = k+1; j < A_nt; ++j) {
                            if (A.tileIsLocal(k, j)) {
                                A.tileUpdateOrigin(k, j);

                                std::set<int> dev_set;
                                A.sub(k+1, A_nt-1, j, j).getLocalDevices(&dev_set);

                                for (auto device : dev_set) {
                                    A.tileUnsetHold(k, j, device);
                                    A.tileRelease(k, j, device);
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }
    A.clearWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization without pivoting.
///
/// Computes an LU factorization without pivoting of a general m-by-n matrix $A$
///
/// The factorization has the form
/// \[
///     A = L U
/// \]
/// where $L$ is lower triangular with unit diagonal elements
/// (lower trapezoidal if m > n), and $U$ is upper triangular
/// (upper trapezoidal if m < n).
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the matrix $A$ to be factored.
///     On exit, the factors $L$ and $U$ from the factorization $A = P L U$;
///     the unit diagonal elements of $L$ are not stored.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, $U(i,i)$ is exactly zero. The
///         factorization has been completed, but the factor $U$ is exactly
///         singular, and division by zero will occur if it is used
///         to solve a system of equations.
///
/// @ingroup gesv_computational
///
template <typename scalar_t>
void getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::getrf_nopiv<Target::HostTask>( A, opts );
            break;

        case Target::HostNest:
            impl::getrf_nopiv<Target::HostNest>( A, opts );
            break;

        case Target::HostBatch:
            impl::getrf_nopiv<Target::HostBatch>( A, opts );
            break;

        case Target::Devices:
            impl::getrf_nopiv<Target::Devices>( A, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrf_nopiv<float>(
    Matrix<float>& A,
    Options const& opts);

template
void getrf_nopiv<double>(
    Matrix<double>& A,
    Options const& opts);

template
void getrf_nopiv< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Options const& opts);

template
void getrf_nopiv< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
