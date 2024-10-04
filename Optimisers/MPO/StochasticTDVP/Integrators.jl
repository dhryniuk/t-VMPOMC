export AdaptiveHeunStep!

function EulerStep!(τ, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    return 0
end

function HeunIntegrate!(y::Array, δ::Float64, N_MC_H::Int64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    
    # k1:
    opt_1 = copy(optimizer, N_MC_H)
    TensorComputeGradient!(opt_1)
    estimators, gradients = MPI_mean!(opt_1, mpi_cache)
    if mpi_cache.rank == 0
        Finalize!(opt_1)
        Reconfigure!(opt_1, estimators, gradients)
    end
    MPI.Bcast!(opt_1.optimizer_cache.∇, 0, mpi_cache.comm)
    k1 = opt_1.optimizer_cache.∇

    # k2:
    opt_2 = copy(optimizer, N_MC_H)
    opt_2.mpo.A += δ*k1
    NormalizeMPO!(opt_2.params, opt_2)
    TensorComputeGradient!(opt_2)
    estimators, gradients = MPI_mean!(opt_2, mpi_cache)
    if mpi_cache.rank == 0
        Finalize!(opt_2)
        Reconfigure!(opt_2, estimators, gradients)

        #k2 = opt_2.optimizer_cache.∇

        # complete integration step:
        #y += δ/2*(k1+k2)
        #opt_1.mpo.A += y
        #NormalizeMPO!(opt_1.params, opt_1)
        #return y, opt_1
    end
    MPI.Bcast!(opt_2.optimizer_cache.∇, 0, mpi_cache.comm)
    k2 = opt_2.optimizer_cache.∇

    # complete integration step:
    y += δ/2*(k1+k2)
    opt_1.mpo.A += y
    NormalizeMPO!(opt_1.params, opt_1)
    return y, opt_1
end

function S_norm(x::Array, optimizer::Optimizer)
    x = reshape(x, prod(size(x)))
    S = optimizer.optimizer_cache.S
    return real( sqrt(x'*S'*x)/length(x) )
end

function AdaptiveHeunStep!(τ, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    # Single τ step:
    y1 = copy(optimizer.mpo.A)
    y1, _ = HeunIntegrate!(y1, τ, optimizer.sampler.N_MC_Heun, optimizer, mpi_cache)

    # Double τ/2 step:
    y2 = copy(optimizer.mpo.A)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, optimizer, mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, opt, mpi_cache)
    
    #delta = S_norm(y1-y2, opt)/6
    #display(delta)
    delta = norm(y1-y2)/6
    #display(delta)

    epsilon = 0.01
    #τ_adjusted = τ*(epsilon/delta)^(1/3)
    τ_adjusted = τ*min((epsilon/delta)^(1/3),2)

    _, optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer.sampler.N_MC, optimizer, mpi_cache)
    return τ_adjusted, optimizer
end