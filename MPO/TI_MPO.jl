
function trMPO(params::Parameters, sample::Projector, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    trMPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt16 in 1:params.N
        trMPO*=A[:,:,idx(sample,i)]
    end
    return tr(trMPO)::ComplexF64
end

#Left partial MPO products:
function L_MPO_products!(L_set, sample::Projector, mpo::TI_MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 

    A = mpo.A
    L_set[1] = cache.ID
    for i=1:params.N
        mul!(L_set[i+1], L_set[i], @view(A[:,:,idx(sample,i)]))
    end
    return L_set
end

#Right partial MPO products:
function R_MPO_products!(R_set, sample::Projector, mpo::TI_MPO{T},  params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 

    A = mpo.A
    R_set[1] = cache.ID
    for i=params.N:-1:1
        mul!(R_set[params.N+2-i], @view(A[:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

function normalize_MPO!(params::Parameters, A::Array{<:Complex{<:AbstractFloat},3})

    MPO=(A[:,:,1]+A[:,:,4])^params.N
    return A./=tr(MPO)^(1/params.N)
end

"""
#Computes the tensor of derivatives of variational parameters: 
function ∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    cache.∂ = zeros(T, params.χ, params.χ, 4)
    for m::UInt16 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt16=1:params.χ, j::UInt16=1:params.χ
            @inbounds cache.∂[i,j,idx(sample,m)] += cache.B[j,i]
        end
    end
    return cache.∂
end
"""

#Computes the tensor of derivatives of variational parameters: 
function ∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 

    cache.∂ = zeros(T, params.χ, params.χ, 4)
    @inbounds for m=1:params.N
        mul!(cache.B, R_set[params.N+1-m], L_set[m])
        @simd for i=1:params.χ
            for j=1:params.χ
                cache.∂[i, j, idx(sample,m)] += cache.B[j, i]
            end
        end
    end
    return cache.∂
end

function conj_∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::TI_MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    cache.∂ = zeros(T, params.χ, params.χ, 4)
    for m::UInt16 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt16=1:params.χ, j::UInt16=1:params.χ
            @inbounds cache.∂[i,j,idx(sample,m)] += conj( cache.B[j,i] )
        end
    end
    return cache.∂
end

#Computes the tensor of derivatives of variational parameters: 
function m∂MPO(sample::Projector, L_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, R_set::Vector{<:Matrix{<:Complex{<:AbstractFloat}}}, params::Parameters, cache::Workspace)
    ∂::Array{eltype(L_set[1]),3} = zeros(eltype(L_set[1]), params.χ, params.χ, 4)
    for m::UInt16 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt16=1:params.χ, j::UInt16=1:params.χ
            @inbounds ∂[i,j,idx(sample,m)] += cache.B[j,i]
        end
    end
    return ∂
end
