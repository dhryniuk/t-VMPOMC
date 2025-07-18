include("../tVMPOMC.jl")
using .tVMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD

mpi_cache = set_mpi()

#Set parameters:
Jx= 0.0 #interaction strength
Jy= 0.0 #interaction strength
Jz= 0.0 #interaction strength
J1= 0.5 #interaction strength
J2= -1.0 #interaction strength
hx= 1.0 #transverse field strength
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
max_τ = parse(Float64,ARGS[9])
#ising_int = "Ising"
ising_int = "CompetingIsing"
#τ = 0.01
τ = 10^(-8)

params = Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)

#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

mpo = MPO("-y", params, mpi_cache)
#mpo2 = MPO("+x", params, mpi_cache)
#mpo.A = 2*mpo.A

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVP(sampler, mpo, l1, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)

#display(optimizer.mpo.A)
#sleep(10000)


basis = generate_bit_basis(N)

# --- plotting additions ---
times_list = Float64[]
mlL2_list = Float64[]
mx_list = Float64[]
my_list = Float64[]
mz_list = Float64[]
Cxx2_list = Float64[]
Cyy2_list = Float64[]
Czz2_list = Float64[]
Cxx3_list = Float64[]
Cyy3_list = Float64[]
Czz3_list = Float64[]
Cxx4_list = Float64[]
Cyy4_list = Float64[]
Czz4_list = Float64[]
# --- end plotting additions ---
S_list = Vector{Vector{Float64}}()

last_iteration_step = 1

#Save parameters to file:
dir = "results/LRIsing_1D_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"
#dir = "results/LRIsing_1D"
if mpi_cache.rank == 0
    if isdir(dir)==true
        cd(dir)
        optimizer = load("optimizer.jld", "optimizer")
        list_of_C = open("C.out", "r")
        last_iteration_step=countlines(list_of_C)+1
    else    
        mkdir(dir)
        cd(dir)
    end
end

MPI.Barrier(mpi_cache.comm)
if mpi_cache.rank != 0
    cd(dir)
end
MPI.bcast(last_iteration_step, 0, mpi_cache.comm)
MPI.bcast(optimizer.τ, 0, mpi_cache.comm)
MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

if mpi_cache.rank == 0 && last_iteration_step == 1
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    mx, my, mz = measure_magnetizations(params, mpo)

    list_of_C = open("C.out", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
    close(list_of_C)

    list_of_obs = open("obs.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz)
    close(list_of_obs)

    Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
    Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
    Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr2.out", "a")
    println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
    close(list_of_obs)

    Cxx3 = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
    Cyy3 = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
    Czz3 = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr3.out", "a")
    println(list_of_obs, Cxx3, ",", Cyy3, ",", Czz3)
    close(list_of_obs)

    Cxx4 = tensor_correlation(1,4,sx,sx,params,mpo) - mx^2
    Cyy4 = tensor_correlation(1,4,sy,sy,params,mpo) - my^2
    Czz4 = tensor_correlation(1,4,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr4.out", "a")
    println(list_of_obs, Cxx4, ",", Cyy4, ",", Czz4)
    close(list_of_obs)

    M_sq = ( modulated_magnetization_TI(0.0, params, mpo, sz) )#^0.5
    M_stag = ( modulated_magnetization_TI(π, params, mpo, sz))#^0.5
    M_mod = ( modulated_magnetization_TI(2π/params.N, params, mpo, sz))#^0.5

    list_of_obs = open("ssf.out", "a")
    println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
    close(list_of_obs)

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)

    # --- plotting additions ---
    push!(times_list, 0.0)
    push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
    push!(mx_list, mx)
    push!(my_list, my)
    push!(mz_list, mz)
    push!(Cxx2_list, Cxx)
    push!(Cyy2_list, Cyy)
    push!(Czz2_list, Czz)
    push!(Cxx3_list, Cxx3)
    push!(Cyy3_list, Cyy3)
    push!(Czz3_list, Czz3)
    push!(Cxx4_list, Cxx4)
    push!(Cyy4_list, Cyy4)
    push!(Czz4_list, Czz4)
    # --- end plotting additions ---

    _, S, _ = find_Schmidt(mpo, params)
    push!(S_list, S)

    list_of_S = open("Schmidt_values.out", "a")
    println(list_of_S, join(S, ","))
    close(list_of_S)
end

MPI.Barrier(mpi_cache.comm)
#Random.seed!(mpi_cache.rank)

current_time::Float64 = parse(Float64, readlines("times.out")[end])
while current_time<T

    global last_iteration_step+=1

    Random.seed!(mpi_cache.rank)
    #EulerStep!(optimizer, mpi_cache)
    t1 = time()
    AdaptiveHeunStepCapped!(max_τ, optimizer, mpi_cache)
    t2 = time()

    if mpi_cache.rank == 0

        iter_time = t2 - t1
        list_of_iter_times = open("iter_times.out", "a")
        println(list_of_iter_times, iter_time)
        close(list_of_iter_times)
    
        #save("optimizer.jld", "optimizer", optimizer)

        mpo = optimizer.mpo

        display(current_time)
        #display(TraceNorm(mpo.A, optimizer))

        _, S, _ = find_Schmidt(mpo, params)
        push!(S_list, S)

        list_of_S = open("Schmidt_values.out", "a")
        println(list_of_S, join(S, ","))
        close(list_of_S)

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)

        mx, my, mz = measure_magnetizations(params, mpo)
        p = tensor_purity(params,mpo)

        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz, ",", p)
        close(list_of_obs)

        Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
        Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
        Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

        list_of_obs = open("corr2.out", "a")
        println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
        close(list_of_obs)

        Cxx3 = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
        Cyy3 = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
        Czz3 = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr3.out", "a")
        println(list_of_obs, Cxx3, ",", Cyy3, ",", Czz3)
        close(list_of_obs)
    
        Cxx4 = tensor_correlation(1,4,sx,sx,params,mpo) - mx^2
        Cyy4 = tensor_correlation(1,4,sy,sy,params,mpo) - my^2
        Czz4 = tensor_correlation(1,4,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr4.out", "a")
        println(list_of_obs, Cxx4, ",", Cyy4, ",", Czz4)
        close(list_of_obs)

        M_sq = ( modulated_magnetization_TI(0.0, 0.0, params, mpo, sz) )#^0.5
        M_stag = ( modulated_magnetization_TI(π, π, params, mpo, sz))#^0.5
        M_mod = ( modulated_magnetization_TI(2π/params.uc_size, 2π/params.uc_size, params, mpo, sz))#^0.5

        list_of_obs = open("ssf.out", "a")
        println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
        close(list_of_obs)

        global current_time += optimizer.τ
        list_of_times = open("times.out", "a")
        println(list_of_times, current_time)
        close(list_of_times)

        ### Positivity:
        a = find_one_site_reduced_smallest_eval(mpo, params)
        display("Smallest eval: ")
        display(a)
        #sleep(1)

        ρ = construct_density_matrix(mpo, params, basis)
        evals, _ = eigen(ρ)
        display("Density matrix smallest eval: ")
        display(minimum(abs.(evals)))

        # --- plotting additions ---
        push!(times_list, current_time)
        push!(mlL2_list, real(optimizer.optimizer_cache.mlL2)/N)
        push!(mx_list, mx)
        push!(my_list, my)
        push!(mz_list, mz)
        push!(Cxx2_list, Cxx)
        push!(Cyy2_list, Cyy)
        push!(Czz2_list, Czz)
        push!(Cxx3_list, Cxx3)
        push!(Cyy3_list, Cyy3)
        push!(Czz3_list, Czz3)
        push!(Cxx4_list, Cxx4)
        push!(Cyy4_list, Cyy4)
        push!(Czz4_list, Czz4)

        if last_iteration_step % 10 == 0

        p = plot(times_list, mlL2_list, yscale=:log10, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mlL2/N", xtick=:auto, ytick=:auto)
        savefig(p, "mlL2.png")
        p = plot(times_list, mx_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mx", xtick=:auto, ytick=:auto)
        savefig(p, "mx.png")
        p = plot(times_list, my_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="my", xtick=:auto, ytick=:auto)
        savefig(p, "my.png")
        p = plot(times_list, mz_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="mz", xtick=:auto, ytick=:auto)
        savefig(p, "mz.png")
        p = plot(times_list, Cxx2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cxx_2", xtick=:auto, ytick=:auto)
        savefig(p, "Cxx_2.png")
        p = plot(times_list, Cyy2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cyy_2", xtick=:auto, ytick=:auto)
        savefig(p, "Cyy_2.png")
        p = plot(times_list, Czz2_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Czz_2", xtick=:auto, ytick=:auto)
        savefig(p, "Czz_2.png")
        p = plot(times_list, Cxx3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cxx_3", xtick=:auto, ytick=:auto)
        savefig(p, "Cxx_3.png")
        p = plot(times_list, Cyy3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cyy_3", xtick=:auto, ytick=:auto)
        savefig(p, "Cyy_3.png")
        p = plot(times_list, Czz3_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Czz_3", xtick=:auto, ytick=:auto)
        savefig(p, "Czz_3.png")
        p = plot(times_list, Cxx4_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cxx_4", xtick=:auto, ytick=:auto)
        savefig(p, "Cxx_4.png")
        p = plot(times_list, Cyy4_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Cyy_4", xtick=:auto, ytick=:auto)
        savefig(p, "Cyy_4.png")
        p = plot(times_list, Czz4_list, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="Czz_4", xtick=:auto, ytick=:auto)
        savefig(p, "Czz_4.png")
        # --- end plotting additions ---

        S_mat = transpose(reduce(hcat, S_list))
        p = plot(times_list, S_mat, dpi=300, size=(600,400), margin=10Plots.mm, xlabel="Time", ylabel="S", xtick=:auto, ytick=:auto, yscale=:log10, ylims=(1e-8, 1.0))
        savefig(p, "sd.png")
        end

        #save("optimizer_backup.jld", "optimizer", optimizer)
    end

    #sleep(10)

    GC.gc()
end

MPI.Barrier(mpi_cache.comm)

error("Exiting")