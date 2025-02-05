function IsingInteractionEnergy(ising_op::Ising, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
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

    return -1.0im*params.J1*l_int
end

function IsingInteractionEnergy(ising_op::LongRangeIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
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
    return -1.0im*params.J1*l_int/ising_op.Kac_norm
end

function IsingInteractionEnergy(ising_op::CompetingIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params

    ie::T=0.0

    l_int_ket::T = 0.0
    l_int_bra::T = 0.0
    l_int::T = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^ising_op.α1
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    ie += -1.0im*params.J1*l_int#/ising_op.Kac_norm1

    l_int_ket = 0.0
    l_int_bra = 0.0
    l_int = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^ising_op.α2
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    ie += -1.0im*params.J2*l_int#/ising_op.Kac_norm2

    return ie
end

function IsingInteractionEnergy(ising_op::SquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
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

    return -1.0im*params.J1*l_int
end

function IsingInteractionEnergy(ising_op::CompetingSquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A
    params = optimizer.params
    N = params.N
    L1 = params.uc_size
    L2 = N ÷ L1
    
    @assert N == L1 * L2 "N must equal L1 * L2 for a rectangular lattice"

    l_int_1::T = 0
    l_int_2::T = 0

    for i::Int64 in 1:N
        for j::Int64 in i+1:N
            # Compute x and y coordinates for the rectangular lattice
            x1, y1 = divrem(i-1, L2)
            x2, y2 = divrem(j-1, L2)

            # Compute distance considering periodic boundary conditions
            dx = min(abs(x1 - x2), L1 - abs(x1 - x2))  # x-distance
            dy = min(abs(y1 - y2), L2 - abs(y1 - y2))  # y-distance
            r = sqrt(dx^2 + dy^2)  # Euclidean distance

            # Contribution
            l_int_ket = (2 * sample.ket[i] - 1) * (2 * sample.ket[j] - 1)
            l_int_bra = (2 * sample.bra[i] - 1) * (2 * sample.bra[j] - 1)

            # Add long-range interaction term with decay r^(-alpha)
            l_int_1 += (l_int_ket - l_int_bra) / r^ising_op.α1
            l_int_2 += (l_int_ket - l_int_bra) / r^ising_op.α2
        end
    end

    return -1.0im * (params.J1 * l_int_1 + params.J2 * l_int_2)
end