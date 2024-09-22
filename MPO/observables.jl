export tensor_magnetization, tensor_purity, tensor_correlation, tensor_cummulant, C2, squared_magnetization, squared_staggered_magnetization, modulated_magnetization, Nagy_structure_factor

"""
function test_mx(ρ0, params::Parameters, mpo::MPO{ComplexF64}, op::Array{ComplexF64})
    A = mpo.A
    B = zeros(ComplexF64,params.χ,params.χ)
    A_reshaped = reshape(A[n,:,:,:],params.χ,params.χ,2,2)
    @tensor B[a,b] := A[a,b,c,d]*ρ0[]
end
"""

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