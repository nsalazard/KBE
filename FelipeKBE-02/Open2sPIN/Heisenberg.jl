# Functions related with the Heisenberg interaction

function compute_Λ(gk::NumOrArray, sp::SpinParams)
    @tullio temp[β,j] := K_tens[a,b,β]*gk[a,b,j]
    @tullio res[α,i] := sp.J[α,β,i,j]*temp[β,j]
    return 0.5im*res
end

"Compute Ω self energies."
function compute_Ω(gk, gs)
    @tullio ok1[a,c,α,i] := K_tens[a,b,α]*gk[b,c,i]
    @tullio ok2[c,a,β,i] := K_tens[c,d,β]*gk[a,d,i]
    @tullio ok[α,β,i] := ok1[a,c,α,i]*ok2[c,a,β,i]
    
    @tullio ok1[a,c,α,i] = K_tens[a,b,α]*gs[b,c,i]
    @tullio ok2[c,a,β,i] = K_tens[c,d,β]*gs[a,d,i]
    @tullio ok[α,β,i] += 0.25*ok1[a,c,α,i]*ok2[c,a,β,i]
    
    @tullio os1[a,c,α,i] := K_tens[a,b,α]*gk[b,c,i]
    @tullio os2[c,a,β,i] := K_tens[c,d,β]*gs[a,d,i]
    @tullio os[α,β,i] := os1[a,c,α,i]*os2[c,a,β,i]
    
    return 0.125im*ok, 0.25im*os
end

"""
    compute_Ms(hist::SystemHistory, sp::SpinParams, t1, t2, h)
    compute_Ms(Ωs::NumOrArray, sp::SpinParams)

Compute Ms(t1, t2) Green functions of the mean-field. Integrals are done with 
trapezoid method with timestep h. If argument is a hist, then integrals are done, 
otherwise, only the term without integral is returned.
"""
function compute_Ms(hist::SystemHistory, sp::SpinParams, t1, t2, h)
    new_M = zeros(ComplexF64, 3,3, sp.N, sp.N)
    @tullio new_M[α,β,i,i] += 4*hist.Ωs[t1,t2][α,β,i]
    
    if t1==t2 return new_M end
    
    func1 = hist.Ωs[t1,:]
    func2 = hist.Ms[:,t2]
      
    list1 = [Array{ComplexF64}(undef, size(new_M)) for i=1:length(func1)]
    list2 = [Array{ComplexF64}(undef, size(new_M)) for i=1:length(func1)]
    
    for k=1:length(list1)
        x = list1[k]
        f = func1[k]
        @tullio x[α,γ,i,j] = 2*f[α,β,i]*sp.J[β,γ,i,j]
    end
    
    for k=1:length(list2)-1
        x = list2[k]
        f1 = list1[k]
        f2 = func2[k]
        @tullio x[α,γ,i,k] = f1[α,β,i,j]*f2[β,γ,j,k]
    end
    
    new_M .+= trapz(list2, t2, t1, h, final_step=false)
    
    inv_factor = I-reshape(vec(permutedims(0.5*h*list1[t1], (1,3,2,4))), 3*sp.N,3*sp.N)
    inv_factor = inv(inv_factor)
    inv_factor = permutedims(reshape(vec(inv_factor), 3,sp.N,3,sp.N), (1,3,2,4))
    
    @tullio res[α,β,i,j] := inv_factor[α,γ,i,k]*new_M[γ,β,k,j]
    
    return res
end

function compute_Ms(Ωs, sp::SpinParams)
    new_M = zeros(ComplexF64, 3,3, sp.N, sp.N)
    @tullio new_M[α,β,i,i] += 4*Ωs[α,β,i]
    return new_M
end

"""
    compute_Mk(hist::SystemHistory, sp::SpinParams, t1, t2, h)
    compute_Mk(Ωs::NumOrArray, sp::SpinParams)

Compute Mk(t1, t2) Green functions of the mean-field. Integrals are done with 
trapezoid method with timestep h. If argument is a hist, then integrals are done, 
otherwise, only the term without integral is returned.
"""
function compute_Mk(hist::SystemHistory, sp::SpinParams, t1, t2, h)
    new_M = zeros(ComplexF64, 3,3, sp.N, sp.N)
    @tullio new_M[α,β,i,i] += 4*hist.Ωk[t1,t2][α,β,i]
    
    if t1==1 return new_M end
    
    func1 = hist.Ωs[t1,:]
    func2 = hist.Mk[:,t2]
      
    list1 = [Array{ComplexF64}(undef, size(new_M)) for i=1:length(func1)]
    list2 = [Array{ComplexF64}(undef, size(new_M)) for i=1:length(func1)]
    
    for k=1:length(list1)
        x = list1[k]
        f = func1[k]
        @tullio x[α,γ,i,j] = 2*f[α,β,i]*sp.J[β,γ,i,j]
    end
    
    for k=1:length(list2)-1
        x = list2[k]
        f1 = list1[k]
        f2 = func2[k]
        @tullio x[α,γ,i,k] = f1[α,β,i,j]*f2[β,γ,j,k]
    end
    
    new_M .+= trapz(list2, 1, t1, h, final_step=false)
    
    inv_factor = I-reshape(vec(permutedims(0.5*h*list1[t1], (1,3,2,4))), 3*sp.N,3*sp.N)
    inv_factor = inv(inv_factor)
    inv_factor = permutedims(reshape(vec(inv_factor), 3,sp.N,3,sp.N), (1,3,2,4))
    
    func1 = hist.Ωk[t1,:]
    func2 = hist.Ms[:,t2]
    
    for k=1:length(list1)
        x = list1[k]
        f = func1[k]
        @tullio x[α,γ,i,j] = 2*f[α,β,i]*sp.J[β,γ,i,j]
    end
    
    for k=1:length(list2)
        x = list2[k]
        f1 = list1[k]
        f2 = func2[k]
        @tullio x[α,γ,i,k] = f1[α,β,i,j]*f2[β,γ,j,k]
    end
    
    new_M .+= -trapz(list2, 1, t2, h, final_step=true)
    
    @tullio res[α,β,i,j] := inv_factor[α,γ,i,k]*new_M[γ,β,k,j]
    
    return res
end

function compute_Mk(Ωk, sp::SpinParams)
    new_M = zeros(ComplexF64, 3,3, sp.N, sp.N)
    @tullio new_M[α,β,i,i] += 4*Ωk[α,β,i]
    return new_M
end
