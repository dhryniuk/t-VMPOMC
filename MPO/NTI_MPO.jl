
function trMPO(params::Parameters, sample::Projector, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    trMPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i::UInt16 in 1:params.N
        trMPO*=A[i,:,:,idx(sample,i)]
    end
    return tr(trMPO)::ComplexF64
end

#Left strings of MPOs:
function L_MPO_products!(L_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    L_set[1] = cache.ID
    for i::UInt16=1:params.N
        mul!(L_set[i+1], L_set[i], @view(A[i,:,:,idx(sample,i)]))
    end
    return L_set
end

#Right strings of MPOs:
function R_MPO_products!(R_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}} 
    A = mpo.A
    R_set[1] = cache.ID
    for i::UInt16=params.N:-1:1
        mul!(R_set[params.N+2-i], @view(A[i,:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

#Computes the tensor of derivatives of variational parameters: 
function ∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    cache.∂ = zeros(T, params.N, params.χ, params.χ, 4)
    for m::UInt16 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt16=1:params.χ, j::UInt16=1:params.χ
            cache.∂[m,i,j,idx(sample,m)] += cache.B[j,i]
        end
    end
    return cache.∂
end

function conj_∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}} 
    cache.∂ = zeros(T, params.N, params.χ, params.χ, 4)
    for m::UInt16 in 1:params.N
        mul!(cache.B,R_set[params.N+1-m],L_set[m])
        for i::UInt16=1:params.χ, j::UInt16=1:params.χ
            cache.∂[m,i,j,idx(sample,m)] += conj(cache.B[j,i])
        end
    end
    return cache.∂
end