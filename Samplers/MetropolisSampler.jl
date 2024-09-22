export MetropolisSampler, Metropolis_sweep_left, MPO_Metropolis_burn_in

function draw_excluded(u::Int8)
    v::Int8 = rand(1:3)
    if v>=u
        v+=1
    end
    return v
end

struct MetropolisSampler
    N_MC::UInt64
    burn::UInt64

    # Sampled data:
    estimators::Vector{ComplexF64}
    gradients::Array{ComplexF64,5}
end

# Constructor:
function MetropolisSampler(N_MC::Int64, burn::Int64, params::Parameters)
    estimators = zeros(ComplexF64, N_MC)
    gradients = zeros(ComplexF64, N_MC, params.uc_size, params.χ, params.χ, 4)
    #gradients = [zeros(ComplexF64, params.uc_size, params.χ, params.χ, 4) for _=1:N_MC]
    return MetropolisSampler(N_MC, burn, estimators, gradients)
end


Base.display(sampler::MetropolisSampler) = begin
    println("\nSampler:")
    println("N_MC\t\t", sampler.N_MC)
    println("burn\t\t", sampler.burn)
end