function MPO(orientation::String, params, mpi_cache)
    Random.seed!(mpi_cache.rank)
    A_init = zeros(ComplexF64, params.uc_size, params.χ, params.χ, 4)
    if orientation=="+x"
        for n in 1:params.uc_size
            A_init[n,:,:,1].=1.0/2
            A_init[n,:,:,2].=1.0/2
            A_init[n,:,:,3].=1.0/2
            A_init[n,:,:,4].=1.0/2
        end
    elseif orientation=="-y"
        for n in 1:params.uc_size
            A_init[n,:,:,1].=1.0/2
            A_init[n,:,:,2].=1.0im/2
            A_init[n,:,:,3].=-1.0im/2
            A_init[n,:,:,4].=1.0/2
        end
    else
        error("Unknown Initial Condition")
    end
    return MPO(A_init)
end