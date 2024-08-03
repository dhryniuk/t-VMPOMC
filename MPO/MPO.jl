export normalize_MPO!

export TI_MPO, MPO, PTI_MPO

abstract type MatrixProductOperator{T<:Complex{<:AbstractFloat}} end

mutable struct TI_MPO{T} <: MatrixProductOperator{T}  
    A::Array{T,3}
end

mutable struct MPO{T} <: MatrixProductOperator{T}
    A::Array{T,4}
end

mutable struct PTI_MPO{T} <: MatrixProductOperator{T}
    A::Array{T,4}
    #uc_size::Int64
end

function Base.similar(mpo::TI_MPO)
    return TI_MPO(similar(mpo.A))
end

function Base.similar(mpo::MPO)
    return MPO(similar(mpo.A))
end

function Base.copy(mpo::TI_MPO)
    return TI_MPO(deepcopy(mpo.A))
end

function Base.copy(mpo::PTI_MPO)
    return PTI_MPO(deepcopy(mpo.A))
end

function Base.copy(mpo::MPO)
    return MPO(deepcopy(mpo.A))
end
