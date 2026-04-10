"""
$(TYPEDEF)

Single-site DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct DMRG{A, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    miniter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    trscheme::TruncationStrategy
    expscheme::TruncationStrategy

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function DMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, alg_eigsolve = (;),
        verbosity = Defaults.verbosity, finalize = Defaults._finalize,
        miniter = 0, trscheme = notrunc(), expscheme = noexpand()
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return DMRG(tol, maxiter, miniter, verbosity, alg_eigsolve′, trscheme, expscheme, finalize)
end

function find_groundstate!(::FiniteChainStyle, ψ, H, alg::DMRG, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            expscheme = updatetruncation(alg.expscheme; iter = iter, current_rank = maximum(map(left_virtualspace, ψ)))
            trscheme = updatetruncation(alg.trscheme; iter = iter)

            zerovector!(ϵs)
            dir = 1
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                pos == length(ψ) && (dir = -1)
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))
                if alg.expscheme isa NoExpand 
                    ψ.AC[pos] = vec 
                elseif dir == 1
                    AL, C = left_orth!(vec; trunc = trscheme)
                    AL, C, ψ.AC[pos + 1] = changebonds_left(AL, C, ψ.AC[pos + 1], expscheme)
                    ψ.AC[pos] = (AL, C)
                elseif dir == -1 
                    C, temp = right_orth!(_transpose_tail(ψ.AC[pos]); trunc = trscheme)
                    C, ψ.AC[pos - 1], temp = changebonds_right(C, ψ.AC[pos - 1], temp, expscheme)
                    ψ.AC[pos] = (C, _transpose_front(temp))
                end
            end
            ϵ = maximum(ϵs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol && iter > alg.miniter
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end

"""
$(TYPEDEF)

Two-site DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct DMRG2{A, S, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    miniter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy
    expscheme::TruncationStrategy

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
# TODO: find better default truncation
function DMRG2(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        miniter = 0, alg_eigsolve = (;), alg_svd = Defaults.alg_svd(), trscheme = notrunc(),
        expscheme = noexpand(), finalize = Defaults._finalize
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return DMRG2(tol, maxiter, miniter, verbosity, alg_eigsolve′, alg_svd, trscheme, expscheme, finalize)
end

function find_groundstate!(::FiniteChainStyle, ψ, H, alg::DMRG2, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            trscheme = updatetruncation(alg.trscheme; iter=iter)
            expscheme = updatetruncation(alg.expscheme; iter = iter, current_rank = maximum(map(left_virtualspace, ψ)))
            zerovector!(ϵs)

            # left to right sweep
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(newA2center; trunc = trscheme, alg = alg.alg_svd)
                al, c = changebonds_left(al, c, expscheme)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # right to left sweep
            for pos in (length(ψ) - 2):-1:1
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(newA2center; trunc = trscheme, alg = alg.alg_svd)
                c, ar = changebonds_right(c, ar, expscheme)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol && iter > alg.miniter
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end
