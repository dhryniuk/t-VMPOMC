export EulerStep!, AdaptiveHeunStep!


export TensorComputeGradient!, EulerIntegrate!, MPI_mean!


function EulerIntegrate!(τ::Float64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    TensorComputeGradient!(optimizer)
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



export testHeunIntegrate!
function testHeunIntegrate!(y::Array, δ::Float64, optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    
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
    #opt_1.mpo.A += y
    #NormalizeMPO!(opt_1.params, opt_1)
    #return y, opt_1
    optimizer.mpo.A += y
    NormalizeMPO!(optimizer.params, optimizer)
    return y, optimizer
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
    #opt_1.mpo.A += y
    #NormalizeMPO!(opt_1.params, opt_1)
    #return y, opt_1
    optimizer.mpo.A += δ/2*(k1+k2)
    #optimizer.mpo.A += y
    NormalizeMPO!(optimizer.params, optimizer)
    return y, optimizer
end


export HeunIntegrate!, working_AdaptiveHeunStep!

function AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    τ = optimizer.τ

    # Single τ step:
    y1 = zeros(size(optimizer.mpo.A))
    y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)

    # Double τ/2 step:
    y2 = zeros(size(optimizer.mpo.A))
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(opt), mpi_cache)

    delta = norm(y1-y2)/6 #/3?
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)
    optimizer.τ = τ_adjusted

    #_, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ, optimizer, mpi_cache)

    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
end

function wAdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ



    # Double τ/2 step:
    y2 = zeros(size(optimizer.mpo.A))
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(opt), mpi_cache)

    # Single τ step:
    y1 = zeros(size(optimizer.mpo.A))
    y1, optimizer = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)

    delta = norm(y1-y2)/6 #/3?
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)#τ*min((optimizer.ϵ_tol/delta)^(1/3),2)
    optimizer.τ = τ_adjusted

    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ, optimizer, mpi_cache)

    #_, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
    #_, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ/2, optimizer, mpi_cache)
end

function working_AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    #=
    # Single τ step:
    #y1 = deepcopy(optimizer.mpo.A)
    y1 = zeros(size(optimizer.mpo.A))
    #y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)
    y1, _ = HeunIntegrate!(y1, 2*τ, deepcopy(optimizer), mpi_cache)
    =#

    # Double τ/2 step:
    #y2 = deepcopy(optimizer.mpo.A)
    y2 = zeros(size(optimizer.mpo.A))
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(opt), mpi_cache)
    #y2, optimizer = HeunIntegrate!(y2, τ, optimizer, mpi_cache)
    #y2, optimizer = HeunIntegrate!(y2, τ, optimizer, mpi_cache)



    # Single τ step:
    #y1 = deepcopy(optimizer.mpo.A)
    y1 = zeros(size(optimizer.mpo.A))
    y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)
    #y1, optimizer = HeunIntegrate!(y1, τ, optimizer, mpi_cache)

   

    delta = norm(y1-y2)/6 #/3?
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)#τ*min((optimizer.ϵ_tol/delta)^(1/3),2)
    #τ_adjusted = τ*min((optimizer.ϵ_tol/delta)^(1/3),1.1)

    #_, optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer, mpi_cache)
    #optimizer = opt
    #display(optimizer.τ)
    #sleep(1)
    optimizer.τ = τ_adjusted
    #display(optimizer.τ)
    #sleep(6)

    _, optimizer = HeunIntegrate!(optimizer.mpo.A, optimizer.τ, optimizer, mpi_cache)

    #display(optimizer.mpo.A)

    ## PROBLEM DUE TO UNNORMALIZED MPO y2 ! -> use optimizer.mpo.A instead

    #_, optimizer = HeunIntegrate!(optimizer.mpo.A, τ_adjusted, optimizer, mpi_cache)
    #_, optimizer = HeunIntegrate!(optimizer.mpo.A, τ_adjusted, optimizer, mpi_cache)
end

"""
function working_AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    # Single τ step:
    y1 = deepcopy(optimizer.mpo.A)
    y1, _ = HeunIntegrate!(y1, τ, deepcopy(optimizer), mpi_cache)
    
    # Double τ/2 step:
    y2 = deepcopy(optimizer.mpo.A)
    y2, opt = HeunIntegrate!(y2, τ/2, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, opt, mpi_cache)
   
    delta = norm(y1-y2)/6 #/3?
    τ_adjusted = τ*(optimizer.ϵ_tol/delta)^(1/3)#τ*min((optimizer.ϵ_Heun/delta)^(1/3),2)

    _, optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer, mpi_cache)
    #optimizer.τ = τ_adjusted
end
"""