export tensor_magnetization, tensor_purity


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
