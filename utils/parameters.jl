export Parameters


mutable struct Parameters
    N::Int64
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
    println("N\t\t\t", params.N)
    println("χ\t\t\t", params.χ)
    println("uc_size\t\t", params.uc_size)
    println("Jx\t\t\t", params.Jx)
    println("Jy\t\t\t", params.Jy)
    println("Jz\t\t\t", params.Jz)
    println("J1\t\t\t", params.J1)
    println("J2\t\t\t", params.J2)
    println("hx\t\t\t", params.hx)
    println("hz\t\t\t", params.hz)
    println("γ_l\t\t\t", params.γ)
    println("γ_d\t\t\t", params.γ_d)
    println("α1\t\t\t", params.α1)
    println("α2\t\t\t", params.α2)
end
