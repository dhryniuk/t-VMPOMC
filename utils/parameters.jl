export Parameters


mutable struct Parameters
    N::Int64
    dim_H::Int64
    dim_L::Int64
    χ::Int64
    Jx::Float64
    Jy::Float64
    Jz::Float64
    J1::Float64
    J2::Float64
    hx::Float64
    hz::Float64
    γ::Float64
    γ_d::Float64
    α1::Float64
    α2::Float64
    uc_size::Int64
end

Base.display(params::Parameters) = begin
    println("\nParameters:")
    println("N\t\t", params.N)
    println("dim_H\t\t", params.dim_H)
    println("dim_L\t\t", params.dim_L)
    println("χ\t\t", params.χ)
    println("Jx\t\t", params.Jx)
    println("Jy\t\t", params.Jy)
    println("Jz\t\t", params.Jz)
    println("J1\t\t", params.J1)
    println("J2\t\t", params.J2)
    println("hx\t\t", params.hx)
    println("hz\t\t", params.hz)
    println("γ_l\t\t", params.γ)
    println("γ_d\t\t", params.γ_d)
    println("α\t\t", params.α1)
    println("α\t\t", params.α2)
    println("uc_size\t\t", params.uc_size)
end

#write a constructor that defaults to 0 whenever some parameter is not specified...

function Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)
    return Parameters(N,2^N,2^(2*N),χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)
end
