include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


# test:
# mpirun -np 2 julia Scripts/run_1D_CompetingIsing.jl 4 6 1 500 2.2 0.001 0.0001 0.01 0.0


mpi_cache = set_mpi()

#Set parameters:
Jx= 0.0 #interaction strength
Jy= 0.0 #interaction strength
Jz= 0.0 #interaction strength
J1= 1.0 #interaction strength
J2= parse(Float64,ARGS[9]) #interaction strength
hx= 2.0 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = parse(Int64,ARGS[1]) #number of spins
α1 = 3.0
α2 = 6.0
γ_d = 0

#Set hyperparameters:
χ = parse(Int64,ARGS[2]) #MPO bond dimension
uc_size = parse(Int64,ARGS[3])
N_MC = parse(Int64,ARGS[4]) #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
sweeps = 2
T = parse(Float64,ARGS[5])
ϵ_shift = parse(Float64,ARGS[6])
ϵ_SNR = parse(Float64,ARGS[7])
ϵ_tol = parse(Float64,ARGS[8])
ising_int="CompetingIsing"
τ = 10^(-8)#0.001#10^(-8)
#τ = 0.01

params = Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)

#Define one-body Lindbladian operator:

l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

#Save parameters to file:
dir = "results/1D_Ising_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"
if mpi_cache.rank == 0
    if isdir(dir)==true
        error("Directory already exists")
    end
    mkdir(dir)
    cd(dir)
end

mpo = MPO("z", params, mpi_cache)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVP(sampler, mpo, l1, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)

mlL2_list = []
mx_list = []
my_list = []
mz_list = []
S2_list = []
Cxx_list = []
Cyy_list = []
Czz_list = []
Mzz_sq_list = []
Mzz_stag_list = []
Mzz_mod_list = []
times_list = [0.0]

if mpi_cache.rank == 0
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    Z = real( tensor_purity(params,mpo) )
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
    mz/=uc_size

    Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
    Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
    Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

    M_sq = ( modulated_magnetization_TI(0.0, 0.0, params, mpo, sz) )#^0.5
    M_stag = ( modulated_magnetization_TI(π, π, params, mpo, sz))#^0.5
    M_mod = ( modulated_magnetization_TI(2π/params.uc_size, 2π/params.uc_size, params, mpo, sz))#^0.5

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)

    list_of_C = open("C.out", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
    close(list_of_C)

    list_of_obs = open("obs.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz, ",", Z, ",", S2)
    close(list_of_obs)

    list_of_obs = open("corr.out", "a")
    println(list_of_obs, Cxx - mx^2, ",", Cyy - my^2, ",", Czz - mz^2)
    close(list_of_obs)

    list_of_obs = open("ssf.out", "a")
    println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
    close(list_of_obs)

    push!(mlL2_list, 1)
    push!(mx_list, mx)
    push!(my_list, my)
    push!(mz_list, mz)
    push!(S2_list, S2)
    push!(Cxx_list, Cxx)
    push!(Cyy_list, Cyy)
    push!(Czz_list, Czz)
    push!(Mzz_sq_list, M_sq)
    push!(Mzz_stag_list, M_stag)
    push!(Mzz_mod_list, M_mod)
end


k = 0
optimizer.τ = τ
while times_list[end]<T

    global k+=1

    AdaptiveHeunStep!(optimizer, mpi_cache)

    if mpi_cache.rank == 0

        mpo = optimizer.mpo

        display(optimizer.τ)

        Z = real( tensor_purity(params,mpo) )
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

        Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
        Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
        Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

        M_sq = ( modulated_magnetization_TI(0.0, 0.0, params, mpo, sz) )#^0.5
        M_stag = ( modulated_magnetization_TI(π, π, params, mpo, sz))#^0.5
        M_mod = ( modulated_magnetization_TI(2π/params.uc_size, 2π/params.uc_size, params, mpo, sz))#^0.5
    
        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)
    
        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz, ",", Z, ",", S2)
        close(list_of_obs)
    
        list_of_obs = open("corr.out", "a")
        println(list_of_obs, Cxx - mx^2, ",", Cyy - my^2, ",", Czz - mz^2)
        close(list_of_obs)

        list_of_obs = open("ssf.out", "a")
        println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
        close(list_of_obs)

        push!(times_list, times_list[end]+optimizer.τ)
        list_of_times = open("times.out", "a")
        println(list_of_times, times_list[end])
        close(list_of_times)
        push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
        push!(mx_list, mx)
        push!(my_list, my)
        push!(mz_list, mz)
        push!(S2_list, S2)
        push!(Cxx_list, Cxx)
        push!(Cyy_list, Cyy)
        push!(Czz_list, Czz)
        push!(Mzz_sq_list, M_sq)
        push!(Mzz_stag_list, M_stag)
        push!(Mzz_mod_list, M_mod)
        

        if mod(k,10)==0
            save("optimizer.jld", "optimizer", optimizer)

            p = plot(times_list, mlL2_list, yscale=:log10)
            savefig(p, "mlL2.png")
            p = plot(times_list, mx_list)
            savefig(p, "mx.png")
            p = plot(times_list, my_list)
            savefig(p, "my.png")
            p = plot(times_list, mz_list)
            savefig(p, "mz.png")
            p = plot(times_list, S2_list)
            savefig(p, "S2.png")
            p = plot(times_list, Cxx_list)
            savefig(p, "Cxx.png")
            p = plot(times_list, Cyy_list)
            savefig(p, "Cyy.png")
            p = plot(times_list, Czz_list)
            savefig(p, "Czz.png")
            p = plot(times_list, Mzz_sq_list)
            savefig(p, "Mzz_sq.png")
            p = plot(times_list, Mzz_stag_list)
            savefig(p, "Mzz_stag.png")
            p = plot(times_list, Mzz_mod_list)
            savefig(p, "Mzz_mod.png")
        end
    end

    if mod(k,10)==0
        GC.gc()
    end
end

exit()
