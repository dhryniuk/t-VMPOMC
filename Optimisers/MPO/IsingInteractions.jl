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

#"""
function IsingInteractionEnergy(ising_op::CompetingSquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A
    params = optimizer.params
    N = params.N
    L = isqrt(N)

    l_int_1::T = 0
    l_int_2::T = 0

    for i::Int64 in 1:N
        for j::Int64 in i+1:N
            # Compute x and y coordinates from i and j
            x1, y1 = divrem(i-1, L)
            x2, y2 = divrem(j-1, L)

            # Compute distance between (x1, y1) and (x2, y2) on the torus (periodic boundary conditions)
            dx = min(abs(x1 - x2), L - abs(x1 - x2))  # Minimum distance in x-direction
            dy = min(abs(y1 - y2), L - abs(y1 - y2))  # Minimum distance in y-direction
            r = sqrt(dx^2 + dy^2)  # Euclidean distance on the torus

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

function oldIsingInteractionEnergy(ising_op::CompetingSquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    L = isqrt(params.N)

    l_int_1::T = 0
    l_int_2::T = 0
    for k1::Int64 in 0:L-1
        for j1::Int64 in 1:L
            for k2::Int64 in 0:L-1
                for j2::Int64 in 1:L
                    # To avoid double counting, only consider pairs where (k2, j2) comes after (k1, j1)
                    if (k2 > k1) || (k2 == k1 && j2 > j1)
                        # Compute distance between (j1, k1) and (j2, k2) on the torus (periodic boundary conditions)
                        dx = min(abs(j1 - j2), L - abs(j1 - j2))  # Minimum distance in x-direction
                        dy = min(abs(k1 - k2), L - abs(k1 - k2))  # Minimum distance in y-direction
                        r = sqrt(dx^2 + dy^2)  # Euclidean distance on the torus

                        #println("(",k1,j1,") (",k2,j2,"): ", r)
                        #sleep(1)

                        # Skip interaction if r is 0 (shouldn't happen due to (j1, k1) != (j2, k2) condition)
                        if r > 0
                            # Horizontal contribution
                            l_int_ket = (2*sample.ket[j1+k1*L] - 1) * (2*sample.ket[j2+k2*L] - 1)
                            l_int_bra = (2*sample.bra[j1+k1*L] - 1) * (2*sample.bra[j2+k2*L] - 1)

                            # Add long-range interaction term with decay r^(-alpha)
                            l_int_1 += (l_int_ket - l_int_bra) / r^ising_op.α1
                            l_int_2 += (l_int_ket - l_int_bra) / r^ising_op.α2
                        end
                    end
                end
            end
        end
    end

    #error()

    return -1.0im * ( params.J1 * l_int_1 + params.J2 * l_int_2 )
end
#"""

function IsingInteractionEnergy(ising_op::TriangularIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
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

    return -1.0im*params.J1*l_int
end