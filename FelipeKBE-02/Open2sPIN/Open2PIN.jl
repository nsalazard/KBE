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
export compute_Σ0!, sample_trajectory, hamiltonian_array, compute_hamiltonian!, create_history0

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
    ts = a:dt:b
    θ = deg2rad(tilt_deg)         # polar angle
    traj = [SVector(sin(θ) * cos(rem(ω*t,2π)),
                    sin(θ) * sin(rem(ω*t,2π)),
                    cos(θ)) for t in ts]
    return ts, traj
end

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

    M = [0 0 -t_hop 0;
         0 0 0 -t_hop;
        -t_hop 0 0 0;
         0 -t_hop 0 0]

    # base Hamiltonian (4x4)
    H = Jsc * (traj[1]*kron(I2,σx) +
               traj[2]*kron(I2,σy) +
               traj[3]*kron(I2,σz)) + M

    #ham = Array{ComplexF64}(undef, 4, 4, n)
    for i in 1:n
        ham[:,:,i] .= H
    end
    #return ham
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
        
"Compute RHS of differential equation for vertical tstep of gk."
function rhs_vertical(hist::SystemHistory, hamiltonian, t1, t2, h)   
    @tullio dk[a,b,i] := 2im*hamiltonian[a,d,i]*hist.gk[t1,t2][d,b,i]
    @tullio ds[a,b,i] := 2im*hamiltonian[a,d,i]*hist.gs[t1,t2][d,b,i]
    
    # This is an efficient way of doing convolution-like integrals. First, all values 
    # of the two functions to be convolved are listed.
    
    func1 = hist.Σs[t1,:] 
    func2 = hist.gk[:,t2]
    
    # A list that will contain the elementwise product of the two functions is
    # allocated.
    
    list = [Array{ComplexF64}(undef, size(dk)) for i=1:length(func1)]
    
    # Elementwise product. Note that it is not straightforward to do this with a 
    # reusable function because different quantities will have different indices 
    # and those indices will be contracted differently.
    
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
    
    # Integration using trapezoid method.
    
    dk .+= trapz(list, 1, t1, h)    
    
    func1 = hist.Σk[t1,:] 
    func2 = hist.gs[:,t2]

    #ERROR?? old list?
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
        
    dk .+= -trapz(list, 1, t2, h)
        
    func1 = hist.Σs[t1,:] 
    func2 = hist.gs[:,t2]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
        
    ds .+= trapz(list, t2, t1, h)
    
    @tullio dk_final[a,b,i] := K0[a,c]*dk[c,b,i]
    @tullio ds_final[a,b,i] := K0[a,c]*ds[c,b,i]
    
    return dk_final, ds_final
end

"Compute RHS of differential equation for diagonal tstep of gk"
function rhs_diag(hist::SystemHistory, hamiltonian, t1, h)
    @tullio dk_ver[a,b,i] := 2im*hamiltonian[a,c,i]*hist.gk[t1,t1][c,b,i]
    @tullio dk_hor[a,b,i] := -2im*hist.gk[t1,t1][a,c,i]*hamiltonian[c,b,i]
    
    func1 = hist.Σs[t1,:] 
    func2 = hist.gk[:,t1]
        
    list = [Array{ComplexF64}(undef, size(dk_ver)) for i=1:length(func1)]
    
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
    
    dk_ver .+= trapz(list, 1, t1, h)    
    
    func1 = hist.Σk[t1,:] 
    func2 = hist.gs[:,t1]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
        
    dk_ver .+= -trapz(list, 1, t1, h)
    
    func1 = hist.gk[t1,:] 
    func2 = hist.Σs[:,t1]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
    
    dk_hor .+= trapz(list, 1, t1, h)    
    
    func1 = hist.gs[t1,:] 
    func2 = hist.Σk[:,t1]
        
    for k=1:length(list)
        x = list[k]
        f1 = func1[k]
        f2 = func2[k]
        @tullio x[a,b,i] = 1im*f1[a,d,i]*f2[d,b,i]
    end
        
    dk_hor .+= -trapz(list, 1, t1, h)
    
    @tullio dk_ver_final[a,b,i] := K0[a,c]*dk_ver[c,b,i]
    @tullio dk_hor_final[a,b,i] := dk_hor[a,c,i]*K0[c,b]
    
    return dk_ver_final + dk_hor_final

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
    fX3(ϵ) = 0.5* im * Γ(ϵ;γ,γc) * (-1 + 2/(1 + exp(βL * (ϵ - μL)))) * exp(-im * ϵ * Δt) / (2π)

    IntegralSL, _ = quadgk(fX3, a, b; rtol=1e-8, atol=1e-10)
    Σk_0[1, 1, 1] = IntegralSL
    Σk_0[2, 2, 1] = IntegralSL

    # Σs (Right lead)

    fX4(ϵ) = 0.5* im  * Γ(ϵ;γ,γc) * (-1 + 2/(1 + exp(βR * (ϵ - μR)))) * exp(-im * ϵ * Δt) / (2π)

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
