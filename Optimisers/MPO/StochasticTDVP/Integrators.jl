export EulerStep!, AdaptiveEulerStep!, AdaptiveHeunStep!


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

        #new_A = similar(A)
        #new_A = A + τ*∇
        #A = new_A
        #optimizer.mpo.A = A
        
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

function AdaptiveEulerStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    τ = optimizer.τ
    
    _, x1 = EulerIntegrate!(τ, deepcopy(optimizer), mpi_cache)
    #optimizer, y1 = EulerIntegrate!(τ/2, optimizer, mpi_cache)
    #optimizer, y2 = EulerIntegrate!(τ/2, optimizer, mpi_cache)
    opt1, y1 = EulerIntegrate!(τ/2, deepcopy(optimizer), mpi_cache)
    opt2, y2 = EulerIntegrate!(τ/2, opt1, mpi_cache)

    δ = norm(x1-(y1+y2)/2)#/2

    #δ = S_norm(x1-(y1+y2)/2, opt)/2
    #display(δ)
    #sleep(5)
    #error()
    #display(optimizer.τ)
    
    #optimizer.τ = 0.9*optimizer.ϵ_Heun*τ^2/δ
    optimizer.τ = τ*min((optimizer.ϵ_Heun/δ)^(1/2),1.1)
    #display(optimizer.τ)
    #println("HERE")
    #sleep(5)

    optimizer, _ = EulerIntegrate!(optimizer.τ, deepcopy(optimizer), mpi_cache)

    return optimizer
end


function wrongAdaptiveEulerStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    τ = optimizer.τ
    
    _, x1 = EulerIntegrate!(τ, deepcopy(optimizer), mpi_cache)
    #optimizer, y1 = EulerIntegrate!(τ/2, optimizer, mpi_cache)
    #optimizer, y2 = EulerIntegrate!(τ/2, optimizer, mpi_cache)
    optimizer_copy1 = deepcopy(optimizer)
    optimizer, y1 = EulerIntegrate!(τ/2, optimizer_copy1, mpi_cache)
    optimizer_copy2 = deepcopy(optimizer_copy1)
    optimizer, y2 = EulerIntegrate!(τ/2, optimizer_copy2, mpi_cache)

    δ = 2*norm(x1-(y1+y2)/2)
    #display(δ)
    #sleep(5)
    #error()
    optimizer.τ = τ*(optimizer.ϵ_Heun/δ)^(1/2)

    return optimizer
end

function EulerStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    optimizer, _ = EulerIntegrate!(optimizer.τ, optimizer, mpi_cache)
    return optimizer
end

#=
function EulerStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    TensorComputeGradient!(optimizer)
    estimators, gradients = MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        EulerIntegrate!(optimizer, estimators, gradients)
    end
    MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)
    return optimizer
end
=#




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


export HeunIntegrate!, working_AdaptiveHeunStep!

function AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    # Single τ step:
    y1 = deepcopy(optimizer.mpo.A)
    y1, _ = HeunIntegrate!(y1, τ, optimizer.sampler.N_MC_Heun, deepcopy(optimizer), mpi_cache)
    
    # Double τ/2 step:
    y2 = deepcopy(optimizer.mpo.A)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, deepcopy(opt), mpi_cache)
   
    delta = norm(y1-y2)/6 #/3?
    #τ_adjusted = τ*(optimizer.ϵ_Heun/delta)^(1/3)#τ*min((optimizer.ϵ_Heun/delta)^(1/3),2)
    τ_adjusted = τ*min((optimizer.ϵ_Heun/delta)^(1/3),1.1)

    #_, optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer.sampler.N_MC, optimizer, mpi_cache)
    optimizer = opt
    optimizer.τ = τ_adjusted
end

function working_AdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    # Single τ step:
    y1 = deepcopy(optimizer.mpo.A)
    y1, _ = HeunIntegrate!(y1, τ, optimizer.sampler.N_MC_Heun, deepcopy(optimizer), mpi_cache)
    
    # Double τ/2 step:
    y2 = deepcopy(optimizer.mpo.A)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, deepcopy(optimizer), mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, opt, mpi_cache)
   
    delta = norm(y1-y2)/6 #/3?
    τ_adjusted = τ*(optimizer.ϵ_Heun/delta)^(1/3)#τ*min((optimizer.ϵ_Heun/delta)^(1/3),2)

    _, optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer.sampler.N_MC, optimizer, mpi_cache)
    optimizer.τ = τ_adjusted
end

function oldAdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    # Single τ step:
    y1 = copy(optimizer.mpo.A)
    y1, _ = HeunIntegrate!(y1, τ, optimizer.sampler.N_MC_Heun, optimizer, mpi_cache)

    # Double τ/2 step:
    y2 = copy(optimizer.mpo.A)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, optimizer, mpi_cache)
    y2, opt = HeunIntegrate!(y2, τ/2, optimizer.sampler.N_MC_Heun, opt, mpi_cache)
    
    delta = norm(y1-y2)/6
    τ_adjusted = τ*(optimizer.ϵ_Heun/delta)^(1/3)#τ*min((optimizer.ϵ_Heun/delta)^(1/3),2)

    _, new_optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer.sampler.N_MC, optimizer, mpi_cache)
    optimizer.mpo = new_optimizer.mpo
    optimizer.optimizer_cache.mlL2 = new_optimizer.optimizer_cache.mlL2
    optimizer.τ = τ_adjusted
end

function pAdaptiveHeunStep!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}

    τ = optimizer.τ

    #display(optimizer.τ)
    #sleep(5)

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

    #epsilon = 0.01
    #τ_adjusted = τ*(epsilon/delta)^(1/3)
    τ_adjusted = τ*(optimizer.ϵ_Heun/delta)^(1/3)#τ*min((optimizer.ϵ_Heun/delta)^(1/3),2)

    #display(τ_adjusted)
    #sleep(5)

    _, new_optimizer = HeunIntegrate!(y1, τ_adjusted, optimizer.sampler.N_MC, optimizer, mpi_cache)
    optimizer.mpo = new_optimizer.mpo
    optimizer.optimizer_cache.mlL2 = new_optimizer.optimizer_cache.mlL2
    optimizer.τ = τ_adjusted

    #display(optimizer.τ)
    #sleep(5)
    #return optimizer
end