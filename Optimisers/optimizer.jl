abstract type OptimizerCache end

abstract type Optimizer{T} end

abstract type TDVP{T} <: Optimizer{T} end

mutable struct TDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,4}
    ΔLL::Array{T,4}

    #Sums:
    mlL::T
    mlL2::T
    acceptance::Float64#UInt64

    #Gradient:
    ∇::Array{T,4}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function TDVPCache(A::Array{T,4},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TDVPCache(
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        convert(T, 0),
        convert(T, 0),
        0.0,
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, 4*params.χ^2*params.uc_size, 4*params.χ^2*params.uc_size),
        zeros(T, 4*params.χ^2*params.uc_size)
    )  
    return cache
end

mutable struct TDVPl1{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

export TDVP

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    elseif ising_int=="LRIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    elseif ising_int=="SquareIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, SquareIsing(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    elseif ising_int=="TriangularIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, TriangularIsing(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

mutable struct TDVPl2{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}
    l2::Array{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

export TDVP

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, l2::Array{T}, ϵ::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TDVPl2(mpo, sampler, TDVPCache(mpo.A, params), l1, l2, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

mutable struct TDVP_H{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}
    l2::Array{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

export TDVP_H

function TDVP_H(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, l2::Array{T}, ϵ::Float64, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    optimizer = TDVP_H(mpo, sampler, TDVPCache(mpo.A, params), l1, l2, SquareIsing(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    return optimizer
end





### EDIT LATER:

"""

abstract type ExactTDVP{T} <: Optimizer{T} end

mutable struct TI_ExactTDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    Z::Float64

    #Sums:
    mlL::T

    #Gradient:
    ∇::Array{T,3}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function TI_ExactTDVPCache(A::Array{T,3},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TI_ExactTDVPCache(
        zeros(T, params.χ, params.χ, 4),
        zeros(T, params.χ, params.χ, 4),
        0.0,
        convert(T,0),
        zeros(T,params.χ,params.χ,4),
        zeros(T,4*params.χ^2,4*params.χ^2),
        zeros(T,4*params.χ^2)
    )  
    return cache
end

mutable struct TI_ExactTDVPl1{T<:Complex{<:AbstractFloat}} <: ExactTDVP{T}

    #MPO:
    mpo::TI_MPO{T}

    #Basis:
    basis::Basis

    #Optimizer:
    optimizer_cache::TI_ExactTDVPCache{T}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}

end

function TDVP(basis::Basis, mpo::TI_MPO{T}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TI_ExactTDVPl1(mpo, basis, TI_ExactTDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    elseif ising_int=="LRIsing"
        optimizer = TI_ExactTDVPl1(mpo, basis, TI_ExactTDVPCache(mpo.A, params), l1, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end
"""