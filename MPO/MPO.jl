export normalize_MPO!

export TI_MPO

abstract type MatrixProductOperator{T} end

#abstract type TI_MPO{T} <: MatrixProductOperator{T} end

mutable struct TI_MPO{T} <: MatrixProductOperator{T}
    A::Array{<:T,3}
end

mutable struct MPO{T} <: MatrixProductOperator{T}
    A::Array{Array{<:T,3}}
end

function trMPO(params::Parameters, sample::Projector, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    trMPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt8 in 1:params.N
        trMPO*=A[i][:,:,idx(sample,i)]
    end
    return tr(trMPO)::ComplexF64
end

#Left strings of MPOs:
function L_MPO_strings!(L_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    L_set[1] = cache.ID
    for i::UInt8=1:params.N
        mul!(L_set[i+1], L_set[i], @view(A[:,:,idx(sample,i)]))
    end
    return L_set
end

#Right strings of MPOs:
function R_MPO_strings!(R_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    R_set[1] = cache.ID
    for i::UInt8=params.N:-1:1
        mul!(R_set[params.N+2-i], @view(A[:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

function trMPO(params::Parameters, sample::Projector, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    trMPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt8 in 1:params.N
        trMPO*=A[:,:,idx(sample,i)]
    end
    return tr(trMPO)::ComplexF64
end

#Left strings of MPOs:
function L_MPO_strings!(L_set, sample::Projector, mpo::TI_MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    L_set[1] = cache.ID
    for i::UInt8=1:params.N
        mul!(L_set[i+1], L_set[i], @view(A[:,:,idx(sample,i)]))
    end
    return L_set
end

#Right strings of MPOs:
function R_MPO_strings!(R_set, sample::Projector, mpo::TI_MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    R_set[1] = cache.ID
    for i::UInt8=params.N:-1:1
        mul!(R_set[params.N+2-i], @view(A[:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

function normalize_MPO!(params::Parameters, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    _MPO=(A[:,:,1]+A[:,:,4])^params.N
    A./=tr(_MPO)^(1/params.N)
    mpo.A = A
end

#Computes the tensor of derivatives of variational parameters: 
function ∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    cache.∂ = zeros(T, params.χ, params.χ, 4)
    #display(cache.∂)
    #error()
    for m::UInt8 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        #display(cache.B)
        #error()
        for i::UInt8=1:params.χ, j::UInt8=1:params.χ
            @inbounds cache.∂[i,j,idx(sample,m)] += cache.B[j,i]
        end
    end
    #display(cache.∂)
    #error()
    return cache.∂
end

#Computes the tensor of derivatives of variational parameters: 
function m∂MPO(sample::Projector, L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, 
    R_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::Parameters, cache::Workspace)
    ∂::Array{eltype(L_set[1]),3} = zeros(eltype(L_set[1]), params.χ, params.χ, 4)
    for m::UInt8 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt8=1:params.χ, j::UInt8=1:params.χ
            @inbounds ∂[i,j,idx(sample,m)] += cache.B[j,i]
        end
    end
    return ∂
end
