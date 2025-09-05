module Open2PIN

using LinearAlgebra
using TransmuteDims
using Tullio
using StaticArrays
using QuadGK

include("SystemTypes.jl")

# export SymmetricGreenFunction, AntisymmetricGreenFunction, SingleField
export SpinParams, BathParams
export create_history, extract_components, make_spin_complex
export evolve_pc!
export compute_Σ0!, sample_trajectory, hamiltonian_array, compute_hamiltonian!, create_history0, evolve_pc0!

"Pauli matrices defined as global constants"
const pauli_0 = hcat([1.,0],[0,1.])
const pauli_x = hcat([0,1.],[1.,0])
const pauli_y = hcat([0,1.0im],[-1.0im,0])
const pauli_z = hcat([1.,0],[0,-1.])

"K matrices for Schwinger bosons defined as global constants"
const K0 = kron(pauli_0, pauli_y)
const Kx = kron(pauli_x, pauli_0)
const Ky = -kron(pauli_y, pauli_y)
const Kz = kron(pauli_z, pauli_0)

"K_vec=[Kx,Ky,Kz] defined as global constant"
const K_vec = [Kx,Ky,Kz];
@tullio K_tens_temp[a,b,α] := K_vec[α][a,b];
"K_tens = K^{alpha}_{ab} defined as global constant"
const K_tens = K_tens_temp

# "Value type 1 for vertical, 2 for diagonal, used in evolution algorithms"
# const ver = Val{1}()
# const diag = Val{2}()

"Type that holds all expectation values of the system and the evolution times."
struct SystemHistory
    gk::AbstractExpectationValue
    gs::AbstractExpectationValue
    Σk::AbstractExpectationValue
    Σs::AbstractExpectationValue
    times::Vector{<: Real}
end

"Holds spin parameters"
mutable struct SpinParams
    "Number of spins"
    N::Int
    "Heisenberg exchange"
    J::NumOrArray
    "External field"
    B::Vector
end

"In-place computation of the Hamiltonian tensor at time t"
function compute_hamiltonian!(out, hist::SystemHistory, sp::SpinParams, t)
    # Field-term comes first because it redefines all elements
    @tullio out[a,b,i] = 0.25*hist.Λ[t][α,i]*K_tens[a,b,α]
    @tullio out[a,b,i] += 0.25*sp.B[α]*K_tens[a,b,α]
end

# NEW Hamiltonian

"""
Return an array of spin unit vectors for times t in [a, b] with step dt.
"""
function sample_trajectory(; ω::Real, dt::Real=0.1, a::Real=0.0, b::Real=1.0, tilt_deg::Real=10)
    ts = a:dt:b+1
    θ = deg2rad(tilt_deg)         # polar angle
    traj = [SVector(sin(θ) * cos(rem(ω*t,2π)),
                    sin(θ) * sin(rem(ω*t,2π)),
                    cos(θ)) for t in ts]
    return traj
end

using StaticArrays, LinearAlgebra

# # Rodrigues' rotation formula
# function rotation_matrix(axis::SVector{3,Float64}, φ::Real)
#     n = axis / norm(axis)   # normalize axis
#     nx, ny, nz = n
#     K = @SMatrix [ 0.0 -nz  ny;
#                    nz  0.0 -nx;
#                   -ny  nx  0.0 ]
#     I3 = I(3)
#     return cos(φ)*I3 + (1-cos(φ))*(n*n') + sin(φ)*K
# end

# function sample_trajectory(; ω::Real, dt::Real=0.1, a::Real=0.0, b::Real=1.0,
#                             tilt_deg::Real=10, axis::SVector{3,Float64}=SVector(0.0,0.0,1.0))
#     ts = a:dt:b+1
#     θ = deg2rad(tilt_deg)

#     # choose initial spin vector tilted by θ from +axis
#     # For simplicity: start with tilt in x-z plane
#     init = SVector(sin(θ), 0.0, cos(θ))

#     traj = SVector{3,Float64}[]
#     for t in ts
#         φ = ω*t
#         R = rotation_matrix(axis, φ)
#         push!(traj, R*init)
#     end
#     return traj
# end


"""
Return an array of 4×4 Hamiltonians corresponding to the spin vectors in `traj`.
"""
function hamiltonian_array(; traj::SVector{3,Float64}, Jsc::Real, t_hop::Real)
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    I2 = Matrix(I, 2, 2)

    M = [0 0 -t_hop 0;
        0 0 0 -t_hop;
        -t_hop 0 0 0;
        0 -t_hop 0 0]

    return [Jsc * (traj[1]*kron(I2,σx) +
                   traj[2]*kron(I2,σy) +
                   traj[3]*kron(I2,σz)) + M]
end

function compute_hamiltonian!(ham; traj::SVector{3,Float64}, Jsc::Real, t_hop::Real, n::Int)
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    I2 = Matrix(I, 2, 2)

    M = [-t_hop 0;
        0 -t_hop]

    # base Hamiltonian (4x4)
    H = Jsc * (traj[1]*σx +
               traj[2]*σy +
               traj[3]*σz) 

    for i in 1:n
        for j in 1:n
            if i == j
                ham[:,:,i,j] .= H
            elseif j == i+1 || j == i-1
                ham[:,:,i,j] .= M
            end
        end
    end
end





"""
    trapz(integrand, a, b, h; final_step=true)

Integrates discrete integrand using trapezoid method from index a to index b. It is
optional to add the term corresponding to the last element, since it may appear on both sides of some integral equations.
"""
function trapz(integrand, a, b, h; final_step=true)
    if a==b
        return zero(integrand[a])
    else
        res = integrand[a]/2
        for i=a+1:b-1
            res += integrand[i]
        end
    end
    if final_step
        res += integrand[b]/2
    end
    return res*h
end

# Utility function to print all elements of hist.gk[t1, t2]
function print_all_gk(hist)
    nT = length(hist.gk.ev)
    for t1 in 1:nT
        for t2 in 1:t1
            println("hist.gk[", t1, ",", t2, "] = ", hist.gk[t1, t2])
        end
    end
end

function print_all_Σs(hist)
    nT = length(hist.Σs.ev)
    for t1 in 1:nT
        for t2 in 1:t1
            println("hist.Σs[", t1, ",", t2, "] = ", hist.Σs[t1, t2])
        end
    end
end
        
"Compute RHS of differential equation for vertical tstep of gk."
function rhs_vertical(hist::SystemHistory, hamiltonian, t1, t2, h)

    @tullio dk[a,b,i,j] := -im*hamiltonian[a,d,i,k]*hist.gk[t1,t2][d,b,k,j]
    @tullio ds[a,b,i,j] := -im*hamiltonian[a,d,i,k]*hist.gs[t1,t2][d,b,k,j]

    ####################################
    if any(isnan, dk)
        error("NaN detected in dk at t1=$t1, t2=$t2")
    end
    if any(isnan, ds)
        error("NaN detected in ds at t1=$t1, t2=$t2")
    end
    #####################################
    
    #println("dk after hamiltonian term: ", size(dk))
    # This is an efficient way of doing convolution-like integrals. First, all values 
    # of the two functions to be convolved are listed.

    func1 = hist.Σs[t1,:] 
    func2 = hist.gk[:,t2]

    # A list that will contain the elementwise product of the two functions is
    # allocated.
    list = [zeros(ComplexF64, size(dk)) for i=1:length(func2)]
    #println(length(func1))
    
    # Elementwise product. Note that it is not straightforward to do this with a 
    # reusable function because different quantities will have different indices 
    # and those indices will be contracted differently.
    
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]  #3D
        f2 = func2[k]  #4D
        #println("f1: ", size(f1))
        #println("f2: ", size(f2))

        if any(isnan, f1)
            error("NaN in f1 at k=$k, t1=$t1, t2=$t2")
        end
        if any(isnan, f2)
            error("NaN in f2 at k=$k, t1=$t1, t2=$t2")
        end
        @tullio x[a,b,i,j] = f1[a,d,i]*f2[d,b,i,j]

        # open("debug_tullio.txt", "a") do io
        #     println(io, "f1 (k=$k, t1=$t1, t2=$t2): size=", size(f1))
        #     for aa in axes(f1,1), dd in axes(f1,2), ii in axes(f1,3)
        #         println(io, "f1[", aa, ",", dd, ",", ii, "] = ", f1[aa,dd,ii])
        #     end
        #     println(io, "f2 (k=$k, t1=$t1, t2=$t2): size=", size(f2))
        #     for dd in axes(f2,1), bb in axes(f2,2), ii in axes(f2,3), jj in axes(f2,4)
        #         println(io, "f2[", dd, ",", bb, ",", ii, ",", jj, "] = ", f2[dd,bb,ii,jj])
        #     end
        #     println(io, "x (k=$k, t1=$t1, t2=$t2): size=", size(x))
        #     for aa in axes(x,1), bb in axes(x,2), ii in axes(x,3), jj in axes(x,4)
        #         println(io, "x[", aa, ",", bb, ",", ii, ",", jj, "] = ", x[aa,bb,ii,jj])
        #     end
        # end

        if any(isnan, x)
            error("NaN in x (elementwise product) at k=$k, t1=$t1, t2=$t2")
        end
    end
    
    # Integration using trapezoid method.
    
    dk .+= -1im * (trapz(list, 1, t1, h) )  
    ####################################
    if any(isnan, dk)
        error("NaN detected in dk at t1=$t1, t2=$t2, step 2")
    end
    #####################################
    
    func1 = hist.Σk[t1,:] 
    func2 = hist.gs[:,t2]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i,j] := f1[a,d,i]*f2[d,b,i,j]
    end
        
    dk .+= 1im*trapz(list, 1, t2, h)

    ####################################
    if any(isnan, dk)
        error("NaN detected in dk at t1=$t1, t2=$t2, step 3")
    end
    #####################################
        
    func1 = hist.Σs[t1,:] 
    func2 = hist.gs[:,t2]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i,j] := f1[a,d,i]*f2[d,b,i,j]
    end
        
    ds .+= -1im * (trapz(list, t2, t1, h))     #NEW
    ####################################
    if any(isnan, ds)
        error("NaN detected in ds at t1=$t1, t2=$t2, step 4")
    end
    #####################################

    return dk, ds
end

function rhs_diag(v_dk)
    h_dk = -conj(permutedims(v_dk, (2, 1, 4, 3)))
    return v_dk + h_dk

end

"Compute RHS of differential equation for diagonal tstep of gk"
function rhs_diag(hist::SystemHistory, hamiltonian, t1, h)
    @tullio dk_ver[a,b,i,j] := -im*hamiltonian[a,d,i,k]*hist.gk[t1,t1][d,b,k,j]
    #@tullio dk_hor[a,b,i,j] :=  conj(-im*hist.gs[t1,t1][a,d,i,k]*hamiltonian[d,b,k,j])
    
    func1 = hist.Σs[t1,:] 
    func2 = hist.gk[:,t1]
        
    list = [zeros(ComplexF64, size(dk_ver)) for i=1:length(func2)]
    
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i,j] = -1im*f1[a,d,i]*f2[d,b,i,j]
    end
    
    dk_ver .+= trapz(list, 1, t1, h)    
    
    func1 = hist.Σk[t1,:] 
    func2 = hist.gs[:,t1]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i,j] = 1im*f1[a,d,i]*f2[d,b,i,j]
    end
        
    dk_ver .+= trapz(list, 1, t1, h)
    
    # func1 = hist.gk[t1,:] 
    # func2 = hist.Σs[:,t1]
        
    # for k=1:length(list)
    #     x = list[k]
    #     f1 = func1[k]
    #     f2 = func2[k]
    #     @tullio x[a,b,i,j] = conj(1im*f1[a,d,i,j]*f2[d,b,i])
    # end
    
    #dk_hor .+= trapz(list, 1, t1, h)    
    
    # func1 = hist.gs[t1,:] 
    # func2 = hist.Σk[:,t1]
        
    # for k=1:length(list)
    #     x = list[k]
    #     f1 = func1[k]
    #     f2 = func2[k]
    #     @tullio x[a,b,i,j] = -conj(1im*f1[a,d,i,j]*f2[d,b,i])
    # end
        
    #dk_hor .+= trapz(list, 1, t1, h)

    dk_hor =  -conj(permutedims(dk_ver, (2, 1, 4, 3)))
    
    return dk_ver + dk_hor
end


function Γ(ϵ; γ=1.0, γc=1.0)
    if abs(ϵ) <= 2γ
        return (γc^2/γ^2) * sqrt(4γ^2 - ϵ^2)
    else
        return 0.0
    end
end
function compute_Σ0(γ, γc, βL, βR, μL, μR, t1, t2, n)  
    a, b = -2γ, 2γ
    Σs_0 = zeros(ComplexF64, 2, 2, n)
    Σk_0 = zeros(ComplexF64, 2, 2, n)

    Δt = t1 - t2
    # Σk (Left lead)
    fX(ϵ) = -im * Γ(ϵ;γ,γc) * exp(-im * ϵ * Δt) * (1 / (2π))

    Integral, _ = quadgk(fX, a, b; rtol=1e-8, atol=1e-10)
    Σs_0[1,1,1] = Integral
    Σs_0[2,2,1] = Integral

    # Σk (Right lead)
    #fX(ϵ) =  -im * (γc^2/(2γ^2)) * (ϵ -im * sqrt(4γ^2 - ϵ^2)) * exp(-im * ϵ * Δt) * (1 / (2π))

    #Integral, _ = quadgk(fX, a, b; rtol=1e-8, atol=1e-10)
    Σs_0[1,1,2] = Integral
    Σs_0[2,2,2] = Integral

    # Σs (Left lead)
    fX3(ϵ) = im * Γ(ϵ;γ,γc) * (-1 + 2/(1 + exp(βL * (ϵ - μL)))) * exp(-im * ϵ * Δt) / (2π)

    IntegralSL, _ = quadgk(fX3, a, b; rtol=1e-8, atol=1e-10)
    Σk_0[1, 1, 1] = IntegralSL
    Σk_0[2, 2, 1] = IntegralSL

    # Σs (Right lead)

    fX4(ϵ) = im  * Γ(ϵ;γ,γc) * (-1 + 2/(1 + exp(βR * (ϵ - μR)))) * exp(-im * ϵ * Δt) / (2π)

    IntegralSR, _ = quadgk(fX4, a, b; rtol=1e-8, atol=1e-10)
    Σk_0[1, 1, 2] = IntegralSR
    Σk_0[2, 2, 2] = IntegralSR

    return Σk_0, Σs_0
end

include("Heisenberg.jl")
include("BosonBath.jl")
include("PredictCorrect.jl")
include("GreenTools.jl")
#include("Adams.jl")

end
