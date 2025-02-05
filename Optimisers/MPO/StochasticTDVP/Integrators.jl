export EulerStep!, AdaptiveHeunStep!, AdaptiveHeunStepCapped!


function EulerIntegrate!(τ::Float64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    estimators, gradients = MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        Finalize!(optimizer)

        Reconfigure!(optimizer, estimators, gradients)
        ∇  = optimizer.optimizer_cache.∇
        
        optimizer.mpo.A += τ * ∇
        NormalizeMPO!(optimizer.params, optimizer)
    end
    MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)
    MPI.Bcast!(optimizer.optimizer_cache.∇, 0, mpi_cache.comm)

    return optimizer, optimizer.optimizer_cache.∇
end

function S_norm(x::Array, optimizer::Optimizer)
    x = reshape(x, prod(size(x)))
    S = optimizer.optimizer_cache.S
    return real( sqrt(x'*S'*x)/length(x) )
end

function EulerStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    optimizer, _ = EulerIntegrate!(optimizer.τ, optimizer, mpi_cache)
    return optimizer
end

function HeunIntegrate!(y::Array, δ::Float64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    # k1:
    opt_1 = deepcopy(optimizer)
    TensorComputeGradient!(opt_1)
    estimators, gradients = MPI_mean!(opt_1, mpi_cache)
    if mpi_cache.rank == 0
        
        Finalize!(opt_1)
        Reconfigure!(opt_1, estimators, gradients)
    end
    MPI.Bcast!(opt_1.optimizer_cache.∇, 0, mpi_cache.comm)
    k1 = opt_1.optimizer_cache.∇

    # k2:
    opt_2 = deepcopy(optimizer)
    opt_2.mpo.A += δ*k1
    NormalizeMPO!(opt_2.params, opt_2)
    TensorComputeGradient!(opt_2)
    estimators, gradients = MPI_mean!(opt_2, mpi_cache)
    if mpi_cache.rank == 0
        Finalize!(opt_2)
        Reconfigure!(opt_2, estimators, gradients)
    end
    MPI.Bcast!(opt_2.optimizer_cache.∇, 0, mpi_cache.comm)
    k2 = opt_2.optimizer_cache.∇

    # complete integration step:
    y += δ/2*(k1+k2)
    optimizer.mpo.A += δ/2*(k1+k2)
    NormalizeMPO!(optimizer.params, optimizer)
    optimizer.optimizer_cache.mlL2 = opt_2.optimizer_cache.mlL2

    optimizer.optimizer_cache.S = opt_2.optimizer_cache.S

    return y, optimizer
end

function AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    τ = optimizer.τ

    # Single τ step:
    y1 = zeros(size(optimizer.mpo.A))
    y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)

    # Double τ/2 step:
    y2 = zeros(size(optimizer.mpo.A))
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, opt, mpi_cache)

    delta = norm(y1-y2)/6
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)
    optimizer.τ = τ_adjusted

    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
end

function AdaptiveHeunStepCapped!(max_τ::Float64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    τ = optimizer.τ

    # Single τ step:
    y1 = zeros(size(optimizer.mpo.A))
    y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)

    # Double τ/2 step:
    y2 = zeros(size(optimizer.mpo.A))
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, opt, mpi_cache)

    delta = norm(y1-y2)/6
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)
    optimizer.τ = min(τ_adjusted, max_τ)

    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
end