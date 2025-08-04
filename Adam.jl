# adaptive_adams.jl

using Printf
using LinearAlgebra  # for norm

"""
    adaptive_adams(f, tspan, y0; hmax=0.1, rtol=1e-6, atol=1e-9)

Solve y' = f(t,y) on tspan = (t0, tfinal) with initial y0 using a 3rd-order
Adams–Bashforth/Moulton predictor–corrector scheme and adaptive step-size.
Returns arrays ts, ys.
"""
function adaptive_adams(f, tspan, y0; hmax=0.1, rtol=1e-6, atol=1e-9)
    t0, tf = tspan
    # Initial step size
    h = min(hmax, (tf - t0)/10)
    t = t0
    y = y0

    ts = [t]
    ys = [y]

    # RK4 for startup
    function rk4_step(t, y, h)
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h,   y + h*k3)
        return y + h*(k1 + 2k2 + 2k3 + k4)/6
    end

    # Two startup steps
    y1 = rk4_step(t, y, h); t1 = t + h
    y2 = rk4_step(t1, y1, h); t2 = t1 + h
    append!(ts, [t1, t2])
    append!(ys, [y1, y2])

    # Store f-evals
    fvals = [f(ts[i], ys[i]) for i in 1:3]  # at t0, t1, t2

    while ts[end] < tf - 1e-12
        t_n2, t_n1, t_n = ts[end-2:end]
        y_n2, y_n1, y_n = ys[end-2:end]
        f_n2, f_n1, f_n = fvals[end-2:end]

        # Predictor (AB3)
        h = min(h, tf - t_n)
        y_pred = y_n + h*(23*f_n - 16*f_n1 + 5*f_n2)/12
        t_pred = t_n + h
        f_pred = f(t_pred, y_pred)

        # Corrector (AM3)
        y_corr = y_n + h*(5*f_pred + 8*f_n - f_n1)/12

        # Error estimate
        err = norm(y_corr - y_pred, Inf)
        tol = atol + rtol * max(norm(y_corr, Inf), norm(y_n, Inf))

        if err <= tol
            # Accept
            push!(ts, t_pred)
            push!(ys, y_corr)
            push!(fvals, f_pred)
            # Update step size
            h = 0.9*h*(tol/err)^(1/4)
            h = min(h, hmax)
        else
            # Reject and reduce
            h = 0.9*h*(tol/err)^(1/4)
        end
    end

    return ts, ys
end

# -- Example: y' = -y, y(0)=1 --
f(t,y) = -y

# Integrate 0→5
ts, ys = adaptive_adams(f, (0.0, 5.0), 1.0; hmax=0.2, rtol=1e-6, atol=1e-9)

# Exact solution and error
exact = [exp(-t) for t in ts]
errors = abs.(ys .- exact)
@printf("Steps taken: %d\n", length(ts))
@printf("Maximum error: %.2e\n", maximum(errors))

println("\n  t    y_num    y_exact   error")
for i in 1:5:length(ts)
    ti = ts[i]
    yi = ys[i]
    ye = exact[i]
    ei = errors[i]
    @printf("%.2f  %8.5f %8.5f %8.2e\n", ti, yi, ye, ei)
end
