function Ising_interaction_energy(ising_op::Ising, sample::Projector, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params

    l_int::T=0
    for j::UInt16 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
    l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
    l_int += l_int_ket-l_int_bra

    return -1.0im*params.J*l_int
end

function Ising_interaction_energy(ising_op::LongRangeIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params

    l_int_ket::T = 0.0
    l_int_bra::T = 0.0
    l_int::T = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^ising_op.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return -1.0im*params.J*l_int/ising_op.Kac_norm
end

function Ising_interaction_energy(ising_op::SquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    L = isqrt(params.N)

    l_int::T=0
    for k::UInt16 in 0:L-1
        for j::UInt16 in 1:L

            #Horizontal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+k*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+k*L]-1)
            l_int += l_int_ket-l_int_bra

            #Vertical:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[j+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[j+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

        end
    end

    return -1.0im*params.J*l_int
end

function Ising_interaction_energy(ising_op::TriangularIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    L = isqrt(params.N)

    l_int::T=0
    for k::UInt16 in 0:L-1
        for j::UInt16 in 1:L

            #Horizontal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+k*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+k*L]-1)
            l_int += l_int_ket-l_int_bra

            #Vertical:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[j+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[j+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

            #Diagonal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

        end
    end

    return -1.0im*params.J*l_int
end