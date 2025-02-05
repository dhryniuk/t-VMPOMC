export tensor_magnetization, tensor_purity, tensor_correlation, tensor_cummulant, C2, squared_magnetization, squared_staggered_magnetization, modulated_magnetization, Nagy_structure_factor


function tensor_magnetization(site, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    A = mpo.A
    B = zeros(ComplexF64,params.χ,params.χ)
    B += diagm(ones(params.χ))
    for i in 1:params.N
        n = mod1(i, params.uc_size)
        A_reshaped = reshape(A[n,:,:,:],params.χ,params.χ,2,2)
        if i==site
            @tensor B[a,b] := B[a,e]*A_reshaped[e,b,c,d]*op[c,d]
        else
            @tensor B[a,b] := B[a,e]*A_reshaped[e,b,c,c]
        end
    end
    return @tensor B[a,a]
end

function tensor_purity(params::Parameters, mpo::MPO{ComplexF64})
    A = mpo.A
    A = reshape(A,params.uc_size,params.χ,params.χ,2,2)
    ms = A[1,:,:,:,:]
    B = rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = ms[a,b,f,e]*ms[u,v,e,f]#conj(A[a,b,e,f])*A[u,v,e,f]
    C=deepcopy(B)
    for i in 2:params.N
        n = mod1(i, params.uc_size)
        ms = A[n,:,:,:,:]
        @tensor C[a,b,u,v] = ms[a,b,f,e]*ms[u,v,e,f]
        @tensor B[a,b,u,v] := B[a,c,u,d]*C[c,b,d,v]
    end
    return @tensor B[a,a,u,u]
end

function tensor_correlation(site1::Int64, site2::Int64, op1::Array{ComplexF64}, op2::Array{ComplexF64}, params::Parameters, mpo::MPO{ComplexF64})
    if site1==site2
        error("Tensor correalations site indices are the same!")
    end
    A = mpo.A
    B = zeros(ComplexF64,params.χ,params.χ)
    B += diagm(ones(params.χ))
    for i in 1:params.N
        n = mod1(i, params.uc_size)
        A_reshaped = reshape(A[n,:,:,:],params.χ,params.χ,2,2)
        if i==site1
            @tensor B[a,b] := B[a,e]*A_reshaped[e,b,c,d]*op1[c,d]
        elseif i==site2
            @tensor B[a,b] := B[a,e]*A_reshaped[e,b,c,d]*op2[c,d]
        else
            @tensor B[a,b] := B[a,e]*A_reshaped[e,b,c,c]
        end
    end
    return real( @tensor B[a,a] )
end

export nn_tensor_correlation_2D

function nn_tensor_correlation_2D(op1::Array{ComplexF64}, params::Parameters, mpo::MPO{ComplexF64})
    A = mpo.A
    B = zeros(ComplexF64,params.χ,params.χ)
    B += diagm(ones(params.χ))
    site0 = 1      
    site1 = 2      
    site2 = params.uc_size + 1     
    sites = [site1,site2]
    corr = 0.0
    for site in sites
        corr += 2*tensor_correlation(site0, site, op1, op1, params, mpo)
    end
    return corr/4
end

function tensor_cummulant(site1::Int64, site2::Int64, op1::Array{ComplexF64}, op2::Array{ComplexF64}, params::Parameters, mpo::MPO{ComplexF64})
    return real( tensor_correlation(site1, site2, op1, op2, params, mpo) )
end

function C2(op1::Array{ComplexF64}, op2::Array{ComplexF64}, params::Parameters, mpo::MPO{ComplexF64})
    c2 = 0
    for i in 1:params.N
        for j in 1:params.N
            if i!=j
                c2 +=  tensor_correlation(i, j, op1, op2, params, mpo)
            else
                c2 += 1
            end
        end
    end
    return real( c2/params.N^2 )
end

function squared_magnetization(params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    m::ComplexF64 = 0
    for i in 1:params.N
        for j in 1:params.N
            if i == j
                m += tensor_magnetization(i, params, mpo, id)
            else
                m += tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end
    return abs( real( m / (params.N^2) ) )
end

function squared_staggered_magnetization(params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    m::ComplexF64 = 0
    for i in 1:params.N
        for j in 1:params.N
            if i == j
                m += tensor_magnetization(i, params, mpo, id)
            else
                m += (-1)^(i+j) * tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end
    return abs( real( m / (params.N^2) ) )
end

function modulated_magnetization(phase::Float64, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    m::ComplexF64 = 0
    for i in 1:params.N
        for j in 1:params.N
            if i == j
                m += tensor_magnetization(i, params, mpo, id)
            else
                m += exp(1im*phase*(j-i)) * tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end
    return abs( real( m / (params.N^2) ) )
end

export modulated_magnetization_TI

function modulated_magnetization_TI(q::Number, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    m::ComplexF64 = 0
    for i in 1:params.N#1
        for j in 1:params.N
            if i == j
                m += tensor_magnetization(i, params, mpo, id)
            else
                m += exp(1im*q*(j-i)) * tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end
    return abs(real(m/params.N^2)) #abs(real(m/params.N))
end

function modulated_magnetization_TI(qx::Number, qy::Number, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    L = isqrt(params.N) # Assuming a square lattice with LxL sites
    S_q::ComplexF64 = 0

    # Loop over all lattice points using linear indices
    for i in 1:L^2
        for j in 1:L^2
            # Calculate 2D coordinates from linear indices
            i_x, i_y = divrem(i-1, L) .+ 1
            j_x, j_y = divrem(j-1, L) .+ 1

            if i == j
                S_q += tensor_magnetization(i, params, mpo, id)
            else
                # Phase factor for 2D: depends on the relative positions (i_x, i_y) and (j_x, j_y)
                delta_x = j_x - i_x
                delta_y = j_y - i_y
                
                # Apply phase modulation
                phase_factor = exp(1im * (qx*delta_x + qy*delta_y))
                S_q += phase_factor * tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end

    return real(S_q)/L^4
end

function Nagy_structure_factor(phase::Float64, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    m::ComplexF64 = 0
    for i in 1:params.N
        for j in 1:params.N
            if i != j
                m += exp(1im*phase*(j-i)) * tensor_correlation(i, j, op, op, params, mpo)
            end
        end
    end
    return abs( real( m / (params.N*(params.N-1)) ) )
end