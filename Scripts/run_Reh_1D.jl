include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Set parameters:
Jx= 2.0 #interaction strength
Jy= 0.0 #interaction strength
Jz= 1.0 #interaction strength
J1= 0.0 #interaction strength
J2= 0.0 #interaction strength
hx= 0.0 #transverse field strength
hz= 1.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = parse(Int64,ARGS[1]) #number of spins
α1 = 9999.0
α2 = 9999.0
γ_d = 0

#Set hyperparameters:
χ = parse(Int64,ARGS[2]) #MPO bond dimension
uc_size = parse(Int64,ARGS[3])
N_MC = parse(Int64,ARGS[4]) #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
δ = parse(Float64,ARGS[5]) #step size
N_iterations = parse(Int64,ARGS[6])
ϵ = parse(Float64,ARGS[7])
ϵ_SNR = parse(Float64,ARGS[8])
ising_int="Ising"

params = Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)

#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)
l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx, sx) + Jy*make_two_body_Lindblad_Hamiltonian(sy, sy) + Jz*make_two_body_Lindblad_Hamiltonian(sz, sz)
l2 = reshape(l2, 4,4,4,4)

#Save parameters to file:
dir = "results/Reh_1D_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"
if mpi_cache.rank == 0
    if isdir(dir)==true
        error("Directory already exists")
    end
    mkdir(dir)
    cd(dir)
end

Random.seed!(mpi_cache.rank)
A_init = zeros(ComplexF64, uc_size,χ,χ,4)
for n in 1:uc_size
    A_init[n,:,:,1].=1.0/2
    A_init[n,:,:,2].=1.0im/2#1.0/2
    A_init[n,:,:,3].=-1.0im/2#1.0/2
    A_init[n,:,:,4].=1.0/2
end
mpo = MPO(A_init)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, params)
optimizer = TDVP(sampler, mpo, l1, l2, ϵ, ϵ_SNR, params, ising_int)
NormalizeMPO!(params, optimizer)

mlL2_list = []
mx_list = []
my_list = []
mz_list = []
S2_list = []
Cxx_list = []
Czz_list = []
C2x_list = []
C2z_list = []
CX_list = []
Mx_stag_list = []
Mx_sq_list = []
Mx_mod_list = []
Mz_stag_list = []
Mz_sq_list = []
Mz_mod_list = []

if mpi_cache.rank == 0
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    #display(optimizer)
    println("\nN_iter\t", N_iterations)
    println("δ\t\t", δ)
    close(list_of_parameters)

    Z = tensor_purity(params,mpo)
    S2 = real( -log(Z)/N )
    mx = 0.0
    my = 0.0
    mz = 0.0
    for n in 1:uc_size
        global mx += real.( tensor_magnetization(n,params,mpo,sx) )
        global my += real.( tensor_magnetization(n,params,mpo,sy) )
        global mz += real.( tensor_magnetization(n,params,mpo,sz) )
    end
    mx/=uc_size
    my/=uc_size
    mz/=uc_size

    Cxx = [tensor_cummulant(1, j, sx, sx, params, mpo) for j=2:params.N÷2+1]
    Czz = [tensor_cummulant(1, j, sz, sz, params, mpo) for j=2:params.N÷2+1]

    Mx_sq = ( squared_magnetization(params, mpo, sx) )^0.5
    Mx_stag = ( squared_staggered_magnetization(params, mpo, sx) )^0.5
    Mx_mod = ( modulated_magnetization(2π/params.N, params, mpo, sx) )^0.5
    Mz_sq = ( squared_magnetization(params, mpo, sz) )^0.5
    Mz_stag = ( squared_staggered_magnetization(params, mpo, sz) )^0.5
    Mz_mod = ( modulated_magnetization(2π/params.N, params, mpo, sz) )^0.5

    list_of_C = open("C.out", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
    close(list_of_C)

    list_of_obs = open("obs.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz, ",", Z, ",", S2)
    close(list_of_obs)

    list_of_obs = open("S_XX.out", "a")
    println(list_of_obs, Mx_sq, ",", Mx_stag, ",", Mx_mod)
    close(list_of_obs)

    list_of_obs = open("S_ZZ.out", "a")
    println(list_of_obs, Mz_sq, ",", Mz_stag, ",", Mz_mod)
    close(list_of_obs)

    list_of_obs = open("CXX.out", "a")
    println(list_of_obs, Cxx)
    close(list_of_obs)

    list_of_obs = open("CZZ.out", "a")
    println(list_of_obs, Czz)
    close(list_of_obs)
end



for k in 1:N_iterations

    #local_estimators = []
    #gradients = []

    #Optimize MPO:
    TensorComputeGradient!(optimizer)
    #TensorComputeGradient!(optimizer, local_estimators, gradients)
    estimators, gradients = MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        Optimize!(optimizer, δ, estimators, gradients)
        #Optimize!(optimizer, δ, optimizer.sampler.estimators, optimizer.sampler.gradients)
        #Optimize!(optimizer)
    end
    MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

    if mpi_cache.rank == 0

        Z = tensor_purity(params,mpo)
        S2 = real( -log(Z)/N )
        mx=0.0
        my=0.0
        mz=0.0
        for n in 1:uc_size
            mx += real.( tensor_magnetization(n,params,mpo,sx) )
            my += real.( tensor_magnetization(n,params,mpo,sy) )
            mz += real.( tensor_magnetization(n,params,mpo,sz) )
        end
        mx/=uc_size
        my/=uc_size
        mz/=uc_size

        Cxx = [tensor_cummulant(1, j, sx, sx, params, mpo) for j=2:params.N÷2+1]
        Czz = [tensor_cummulant(1, j, sz, sz, params, mpo) for j=2:params.N÷2+1]

        Mx_sq = ( squared_magnetization(params, mpo, sx) )^0.5
        Mx_stag = ( squared_staggered_magnetization(params, mpo, sx) )^0.5
        Mx_mod = ( modulated_magnetization(2π/params.N, params, mpo, sx) )^0.5
        Mz_sq = ( squared_magnetization(params, mpo, sz) )^0.5
        Mz_stag = ( squared_staggered_magnetization(params, mpo, sz) )^0.5
        Mz_mod = ( modulated_magnetization(2π/params.N, params, mpo, sz) )^0.5

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)

        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz, ",", Z, ",", S2)
        close(list_of_obs)

        list_of_obs = open("S_XX.out", "a")
        println(list_of_obs, Mx_sq, ",", Mx_stag, ",", Mx_mod)
        close(list_of_obs)

        list_of_obs = open("S_ZZ.out", "a")
        println(list_of_obs, Mz_sq, ",", Mz_stag, ",", Mz_mod)
        close(list_of_obs)

        list_of_obs = open("CXX.out", "a")
        println(list_of_obs, Cxx)
        close(list_of_obs)

        list_of_obs = open("CZZ.out", "a")
        println(list_of_obs, Czz)
        close(list_of_obs)

        push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
        push!(mx_list, mx)
        push!(my_list, my)
        push!(mz_list, mz)
        push!(S2_list, S2)
        push!(Mx_sq_list, Mx_sq)
        push!(Mx_stag_list, Mx_stag)
        push!(Mx_mod_list, Mx_mod)
        push!(Mz_sq_list, Mz_sq)
        push!(Mz_stag_list, Mz_stag)
        push!(Mz_mod_list, Mz_mod)
        push!(Cxx_list, Cxx)
        push!(Czz_list, Czz)
        if mod(k,10)==0
            p = plot(mlL2_list, yscale=:log10)
            savefig(p, "mlL2.png")
            p = plot(my_list)
            savefig(p, "my.png")
            p = plot(mx_list)
            savefig(p, "mx.png")
            p = plot(mz_list)
            savefig(p, "mz.png")
            p = plot(S2_list)
            savefig(p, "S2.png")
            p = plot(Mx_sq_list)
            savefig(p, "Mx_sq.png")
            p = plot(Mx_stag_list)
            savefig(p, "Mx_stag.png")
            p = plot(Mx_mod_list)
            savefig(p, "Mx_mod.png")
            p = plot(Mz_sq_list)
            savefig(p, "Mz_sq.png")
            p = plot(Mz_stag_list)
            savefig(p, "Mz_stag.png")
            p = plot(Mz_mod_list)
            savefig(p, "Mz_mod.png")
            #for j=1:params.N÷2
            #    p = plot(map(v -> v[j], Cxx_list))
            #    savefig(p, "Cxx_$(j).png")
            #    p = plot(map(v -> v[j], Czz_list))
            #    savefig(p, "Czz_$(j).png")
            #end
        end
    end

    if mod(k,10)==0
        GC.gc()
    end
end

exit()