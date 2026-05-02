println("
-----------------------------
|   Correlators & Entropy    |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "correlation length / entropy" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^10])
    H = force_planar(transverse_field_ising())
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_crit = correlation_length(ψ)[1]
    entrop_crit = entropy(ψ)

    H = force_planar(transverse_field_ising(; g = 4))
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_gapped = correlation_length(ψ)[1]
    entrop_gapped = entropy(ψ)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "expectation value / correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

    @test expectation_value(ψ, H) ≈
        expectation_value(ψ, 1 => -2g * S_x()) + expectation_value(ψ, (1, 2) => -4S_z_S_z())

    Z_mpo = MPSKit.add_util_leg(S_z())

    # correlator(ψ, Z, Z, i, d): i=1, distances d=1..4 → O1 at 1, O2 at 2..5
    G = correlator(ψ, Z_mpo, Z_mpo, 1, 1:4)

    # G[d] = <Z[1] Z[1+d]>; compare against expectation_value with 2-site operator
    @test isapprox(G[1], expectation_value(ψ, (1, 2) => S_z_S_z()), atol = 1.0e-2)
    @test isapprox(G[2], expectation_value(ψ, (1, 3) => S_z_S_z()), atol = 1.0e-2)

    # Single-distance scalar call must match the range result.
    @test correlator(ψ, Z_mpo, Z_mpo, 1, 1) ≈ G[1] atol = 1e-10
    @test correlator(ψ, Z_mpo, Z_mpo, 1, 2) ≈ G[2] atol = 1e-10
end

@testset "connected correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

    Z_mpo = MPSKit.add_util_leg(S_z())

    # i=1, distances 1..4 → O2 at 2..5
    G_full = correlator(ψ, Z_mpo, Z_mpo, 1, 1:4)
    G_conn = correlator(ψ, Z_mpo, Z_mpo, 1, 1:4; connected = true)

    # For a translation-invariant InfiniteMPS, <O[i]> = ev for all sites.
    ev = expectation_value(ψ, 1 => S_z())
    @test G_conn ≈ G_full .- abs2(ev)

    # Single-distance scalar call should match the range result.
    @test correlator(ψ, Z_mpo, Z_mpo, 1, 3; connected = true) ≈ G_conn[3]
end

@testset "N-point correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

    Z_mpo = MPSKit.add_util_leg(S_z())
    ev = expectation_value(ψ, 1 => S_z())

    # 3-point with range over last distance: i=1, d1=1 (O2 at 2), d2=1..4 (O3 at 3..6)
    G3_range = correlator(ψ, Z_mpo, Z_mpo, Z_mpo, 1, 1, 1:4)
    @test length(G3_range) == 4
    for (k, d2) in enumerate(1:4)
        @test G3_range[k] ≈ correlator(ψ, Z_mpo, Z_mpo, Z_mpo, 1, 1, d2) atol = 1e-10
    end

    # Tuple interface agrees with small-N dispatcher.
    G3_tuple = correlator(ψ, (Z_mpo, Z_mpo, Z_mpo), (1, 1, 1))
    @test G3_tuple ≈ correlator(ψ, Z_mpo, Z_mpo, Z_mpo, 1, 1, 1) atol = 1e-10

    # 4-point: tuple and small-N dispatcher agree.
    G4 = correlator(ψ, Z_mpo, Z_mpo, Z_mpo, Z_mpo, 1, 1, 1, 1)
    G4_tuple = correlator(ψ, (Z_mpo, Z_mpo, Z_mpo, Z_mpo), (1, 1, 1, 1))
    @test G4 ≈ G4_tuple atol = 1e-10

    # Multi-position output: shape (length(is), length(ds))
    G_mat = correlator(ψ, (Z_mpo, Z_mpo), (1:3, 1:4))
    @test size(G_mat) == (3, 4)
    for (ii, i) in enumerate(1:3), (di, d) in enumerate(1:4)
        @test G_mat[ii, di] ≈ correlator(ψ, Z_mpo, Z_mpo, i, d) atol = 1e-10
    end

    # Connected 3-point: G_conn = G - G2(1,2)*ev - G2(1,3)*ev - G2(2,3)*ev + 2*ev^3
    # Sites: O1 at 1, O2 at 2 (d1=1), O3 at 3 (d2=1)
    G3      = correlator(ψ, Z_mpo, Z_mpo, Z_mpo, 1, 1, 1)
    G3_conn = correlator(ψ, Z_mpo, Z_mpo, Z_mpo, 1, 1, 1; connected = true)
    G2_12   = correlator(ψ, Z_mpo, Z_mpo, 1, 1)   # <Z[1]Z[2]>
    G2_13   = correlator(ψ, Z_mpo, Z_mpo, 1, 2)   # <Z[1]Z[3]>
    G2_23   = correlator(ψ, Z_mpo, Z_mpo, 2, 1)   # <Z[2]Z[3]>
    expected_conn = G3 - G2_12 * ev - G2_13 * ev - G2_23 * ev + 2 * ev^3
    @test G3_conn ≈ expected_conn atol = 1e-8

    # connected=true with multi-site operators: d=2 means O2 starts 2 sites after the last
    # site of O1 (a 2-site operator), so O1 occupies sites 1-2 and O2 starts at site 4.
    G_conn_ms = correlator(ψ, (S_z_S_z(), Z_mpo), (1, 2); connected = true)
    G_full_ms = correlator(ψ, (S_z_S_z(), Z_mpo), (1, 2))
    ev12 = expectation_value(ψ, (1, 2) => S_z_S_z())
    ev4  = expectation_value(ψ, 4 => S_z())
    @test G_conn_ms ≈ G_full_ms - ev12 * ev4 atol = 1e-8
end
