#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t> void test_hesv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    //---------------------
    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Target target = params.target();

    //---------------------
    // mark non-standard output values
    params.time();
    params.gflops();

    if (! run)
        return;

    int64_t Am = n;
    int64_t An = n;

    //---------------------
    // Local values
    const int izero = 0, ione = 1;

    //---------------------
    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    //---------------------
    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    //---------------------
    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    //---------------------
    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::Pivots pivots;

    //---------------------
    // band matrix
    int64_t kl = nb;
    int64_t ku = nb;
    slate::Pivots pivots2;
    auto T = slate::BandMatrix<scalar_t>(n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t> (n, n, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // right-hand-side and solution vectors
    int64_t Bm = n;
    int64_t Bn = n;
    int descB_tst[9], descB_ref[9];
    std::vector<scalar_t> B_ref;

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    B_ref.resize(B_tst.size());
    B_ref = B_tst;
    scalapack_descinit(descB_ref, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);

    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    //---------------------
    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if (check) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (params.routine == "hetrs") {
        slate::hetrf(A, pivots, T, pivots2, H, {
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads}
        });
    }

    //==================================================
    // Run SLATE test.
    // Factor A = LTL^H.
    //==================================================
    double time = libtest::get_wtime();
    if (params.routine == "hetrf") {
        slate::hetrf(A, pivots, T, pivots2, H, {
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads}
        });
    } else if (params.routine == "hetrs") {
        slate::hetrs(A, pivots, T, pivots2, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });
    } else {
        slate::hesv(A, pivots, T, pivots2, H, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });
    }

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    //---------------------
    // compute and save timing/performance
    double gflop;
    if (params.routine == "hetrf")
        gflop = lapack::Gflop<scalar_t>::potrf(n);
    else if (params.routine == "hetrs")
        gflop = lapack::Gflop<scalar_t>::potrs(n, Bn);
    else
        gflop = lapack::Gflop<scalar_t>::posv(n, Bn);
    params.time() = time_tst;
    params.gflops() = gflop / time_tst;

    if (check) {
        if (params.routine == "hetrf") {
            // solve
            slate::hetrs(A, pivots, T, pivots2, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }

        // allocate work space
        std::vector<real_t> worklangeA(std::max(mlocA, nlocA));
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of the orig matrix: || A ||_I
        real_t A_norm = scalapack_plange(norm2str(norm), Am, An, &A_ref[0], ione, ione, descA_ref, &worklangeA[0]);
        // norm of updated rhs matrix: || X ||_I
        real_t X_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= Aref*B_tst
        scalapack_phemm("Left", "Lower",
                        Bm, Bn,
                        scalar_t(-1.0),
                        &A_ref[0], ione, ione, descA_ref,
                        &B_tst[0], ione, ione, descB_tst,
                        scalar_t(1.0),
                        &B_ref[0], ione, ione, descB_ref);

        // || B - AX ||_I
        real_t R_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_ref[0], ione, ione, descB_ref, &worklangeB[0]);

        double residual = R_norm / (n*A_norm*X_norm);
        params.error() = residual;

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_hesv(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hesv_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_hesv_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_hesv_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_hesv_work<std::complex<double>> (params, run);
            break;
    }
}
