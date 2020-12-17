using Printf
using Statistics
using Plots
using Roots

#2.4 Calculate chi
global α = 1/3
global β = 0.98
global z = 1
global σ = 2
global η = 1
l = 0.4
global δ =1
global M = ((z*α*β)/(1-β+β*δ))^(1/(1-α))
global N = z*M^α - δ*M
global χ = (z*(1-α)*M^α)/(l^(σ+η)*N^σ)
global max_iter = 500
global dist_tol = 1E-4
global H_tol = 1E-4

global data_k = zeros(100, 6)
#2.5
# current period Utility
function utility(k,kp,l)
    c = z*k^α*l^(1-α)-kp
    la = χ*l^(1+η)/(1+η)
    if c>0
    return c^(1-σ)/(1-σ)-la
    else
    return -Inf
    end
end

# Steady state values (funciton)
function SS_values()
    M = ((z*α*β)/(1-β+β*δ))^(1/(1-α))
    N = z*M^α - δ*M
    χ = (z*(1-α)*M^α)/(l^(σ+η)*N^σ)
    l_ss = ((z*(1-α)*M^α)/(χ*N^σ))^(1/(σ+η))
    k_ss = M*l_ss
    c_ss = N*l_ss
    y_ss = z*M^α*l_ss
    r_ss = α*z*M^(α-1)
    w_ss = (1-α)*z*M^α
    return l_ss,k_ss,c_ss,y_ss,r_ss,w_ss
end

# Best l given k and kp
function opt_l(k,kp)
    #f(x) = (z*k^α*x^(1-α)-kp)^(-σ)*(1-α)*z*k^α*x^(-α)-χ*x^η
    #l_opt =find_zero(f, [0, 1])
    #left = (χ/((1-α)*z*k^α))^(-1/σ)-z*k^α
    #l_opt = (-kp/left)^(-1/(1-α))
    #println("labor------------------------" )
    itr = 100
    l_aux = zeros(itr)
    l_grid = range(0,1;length=itr)
    for i = 1:itr
        l_aux[i] = utility(k,kp,l_grid[i])
        #println(l_grid[i])
        #println("utility= ", l_aux[i])
        #println("ite: $i" )
    end
    value, place = findmax(l_aux)
    #println("noooooo" )
    #println("best l utility: $value" )
    #println("best l: ", l_grid[place])
    #println("labor------------------------------" )
    return l_grid[place]

end

# VFI with Grid
function Make_K_Grid(n_k)
    # Get SS
    l_ss,k_ss,c_ss,y_ss,r_ss,w_ss = SS_values()
    # Get k_grid
    k_grid = range(1E-5,2*k_ss;length=n_k) ; # Equally spaced grid between 0 and 2*k_ss
    # Return
    return k_grid
end

function VFI_grid_loop(n_k)
    println(" ")
    println("------------------------")
    println("VFI with grid search and loops - n_k=$n_k")
    # Get SS
    l_ss,k_ss,c_ss,y_ss,r_ss,w_ss = SS_values()
    # Get k_grid
    k_grid   = Make_K_Grid(n_k)  # Equally spaced grid between 0 and 2*k_ss
    # Initialize variables for loop
    V_old  = zeros(n_k)  # Initial value, a vector of zeros
    iter   = 1           # Iteration index
    V_dist = 1
    converge = 1        # Initialize distance
    while iter<=max_iter && V_dist>dist_tol
        println("---------------iteration-------------")
        # Update value function
        V_new, G_kp, G_c = T_grid_loop(V_old,k_grid)
        # Update distance and iterations
        V_dist = maximum(abs.(V_new./V_old.-1))
        iter  += 1
        # Update old function
        V_old  = V_new
        # Report progress
        if mod(iter,50)==0
            data_k[:,converge] = V_old
            converge += 1
            println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        end
    end
    # Check solution
    if (V_dist<=dist_tol)
        # Recover value and policy functions
        V, G_kp, G_c = T_grid_loop(V_old,k_grid)
        # Return
        println("VFI with grid search and loops - Completed - n_k=$n_k")
        println("Iterations = $iter and Distance = ",100*V_dist,"%")
        println("------------------------")
        println(" ")
        data_k[:,converge] = V
        return V, G_kp, G_c, k_grid
    else
        println("Error in VFI with loops - Solution not found ")
        return V_old, G_kp, G_c, k_grid
        #error("Error in VFI with loops - Solution not found")
    end
end

# Define function for Value update and policy functions
function T_grid_loop(V_old,k_grid)
    n_k  = length(k_grid)
    V    = zeros(n_k)
    G_kp = fill(0,n_k)
    G_c  = zeros(n_k) #positive
    for i = 1:n_k
        V_aux = zeros(n_k) ; # Empty vector for auxiliary value of V(i,j)
        for j = 1:n_k
            # Evaluate potential value function for combinations of
            # current capital k_i and future capital k_j
            #println("------")
            #println(k_grid[i])
            #println(k_grid[j])
            o_l = opt_l(k_grid[i],k_grid[j])
            #println("i: $i, j: $j")
            V_aux[j] = utility(k_grid[i],k_grid[j],o_l) + β*V_old[j]
            #println("V_aux[$j]", V_aux[j])
            #println(V_aux[j]," ",k_grid[i]," ",k_grid[j]," ",utility(k_grid[i],k_grid[j],z,a,b) + b*V_old[j])
        end
        # Choose maximum value given current capital k_i
        V[i], G_kp[i] = findmax(V_aux)
        #println("V[$i]:", V[i], ", G_kp[$i]:", G_kp[i])
        #println("G_kp[$i]", G_kp[i])
        #println("k_grid[G_kp[$i]]", k_grid[G_kp[i]])
        o_l = opt_l(k_grid[i],k_grid[G_kp[i]])
        #println("Best l for k = k_grid[$i] is ", o_l)
        G_c[i]        = z*k_grid[i]^α*o_l^(1-α) - k_grid[G_kp[i]]
    end
    return V, G_kp, G_c
end

@time V_100, G_kp_100, G_c_100, k_grid_100 = VFI_grid_loop(100)
data = zeros(100, 1)

for i = 1:100
    data[i]= k_grid_100[G_kp_100[i]]
end
#plot policy function
plot(k_grid_100, data,
label = "g(k)", legend = :bottomright, widths = [10], title = "Policy Function")
plot!(k_grid_100,k_grid_100, legend = :bottomright,label="")
plot!([SS_values()[2]], seriestype="vline", label="Steady State")
xlabel!("Capital")
ylabel!("Capital")
#savefig("HW2_5a_policy")

#Value function graph
plot(k_grid_100, [data_k[:,1]],
label = "ite 50", legend = :bottomright, widths = [10], title = "Value Function")
plot!(k_grid_100, [data_k[:,2]],
label = "ite 100", legend = :bottomright, widths = [10])
plot!(k_grid_100, [data_k[:,3]],
label = "ite 150", legend = :bottomright, widths = [10])
plot!(k_grid_100, [data_k[:,4]],
label = "ite 200", legend = :bottomright, widths = [10])
plot!(k_grid_100, [data_k[:,5]],
label = "ite 250", legend = :bottomright, widths = [10])
plot!(k_grid_100, [data_k[:,6]],
label = "limit", legend = :bottomright, widths = [10])
xlabel!("Capital")
ylabel!("Value")
#savefig("HW2_5a_Value")


function VFI_Analytical_Results(n_k,G_kp)
    # Get Grid
    k_grid = Make_K_Grid(n_k)
    # Analytical value function
    # Euler error of numerical policy function on grid
    Euler = zeros(n_k)
    for i=1:n_k
        k   = k_grid[i]
        kp  = k_grid[G_kp[i]]
        kpp = k_grid[G_kp[G_kp[i]]]
        l1 = opt_l(k,kp)
        l2 = opt_l(kp,kpp)
        Euler[i] = Euler_Error(k,kp,kpp,l1,l2)
    end
    # Return
    return Euler
end

function Euler_Error(k,kp,kpp,l1,l2)
    # Return percentage error in Euler equation
    LHS = 1/(z*k^α*l1^(1-α)-kp)^σ
    RHS = β*α*z*kp^(α-1)*l2^(1-α)/(z*kp^α*l2^(1-α)-kpp)^σ
    return (RHS/LHS-1)*100
end

Euler_100 = VFI_Analytical_Results(100,G_kp_100)

#Euler graph
plot(k_grid_100, Euler_100,linetype=:scatter,marker=(:diamond,3),
label = "Error", title= "Euler Equation Error(%)", legend = :bottomright)
xlabel!("Capital")
ylabel!("Percentage Points")
#savefig("HW2_5a_Euler")

println("----------------------------------------------------------------")
println("--Howard's policy Iteration--------------------------------------------------------------")

function VFI_HPI_grid_mat(n_H,n_k)
    println(" ")
    println("------------------------")
    println("VFI with Howard's Policy Iteration - n_k=$n_k")
    # Get SS
    l_ss,k_ss,c_ss,y_ss,r_ss,w_ss = SS_values()
    # Get k_grid
    k_grid   = range(1E-5,2*k_ss;length=n_k) ; # Equally spaced grid between 0 and 2*k_ss
    # Utility matrix

    U_mat = [utility(k_grid[i],k_grid[j],opt_l(k_grid[i],k_grid[j])) for i in 1:n_k, j in 1:n_k]
    # Initialize variables for loop
    V_old  = zeros(n_k) ; # Initial value, a vector of zeros
    iter   = 0          ; # Iteration index
    V_dist = 1          ; # Initialize distance
    while iter<=max_iter && V_dist>dist_tol
        # Update value function
        println("Loop iter= $iter")
        V_new, G_kp, G_c = HT_grid_mat(V_old,U_mat,k_grid,n_H)
        # Update distance and iterations
        V_dist = maximum(abs.(V_new./V_old.-1))
        iter  += 1
        # Update old function
        V_old  = V_new
        # Report progress
        if mod(iter,50)==0
            println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        end
    end
    # Check solution
    if (V_dist<=dist_tol)
        # Recover value and policy functions
        V, G_kp, G_c = T_grid_loop(V_old,k_grid)
        # Return
        println("VFI with  Howard's Policy Iteration - Completed - n_k=$n_k")
        println("Iterations = $iter and Distance = ",100*V_dist,"%")
        println("------------------------")
        println(" ")
        return V, G_kp, G_c, k_grid
    else
        error("Error in VFI with Howard's Policy Iteration - Solution not found")
    end
end

    # Define function for Value update and policy functions
function HT_grid_mat(V_old,U_mat,k_grid,n_H)
    # Get Policy Function
    n_k    = length(V_old)
    V,G_kp = findmax( U_mat .+ β*repeat(V_old',n_k,1) , dims=2 )
    V_old  = V
    # "Optimal" U for Howard's iteration
        U_vec = U_mat[G_kp]
    # Howard's policy iteration
    # G_kp is a Cartesian Index
    for i=1:n_H
        V = U_vec .+ β*repeat(V_old',n_k,1)[G_kp]
        if maximum(abs.(V./V_old.-1))<=H_tol
            break
        end
        V_old = V
    end
    # Recover Policy Functions
    G_kp   = [G_kp[i][2] for i in 1:n_k] # G_kp is a Cartesian index
    opl = zeros(n_k)
    for i=1:n_k
        opl[i] = opt_l(k_grid[i],k_grid[G_kp[i]])
   end

    G_c    = (z*(k_grid.^α).*opl.^(1-α))  .- k_grid[G_kp]
    # Return output
    return V, G_kp, G_c
end

#@time V_50_H, G_kp_50_H, G_c_50_H, k_grid_50_H = VFI_HPI_grid_mat(50,50)
#@time V_200_H, G_kp_200_H, G_c_200_H, k_grid_200_H = VFI_HPI_grid_mat(200,200)
@time V_500_H, G_kp_500_H, G_c_500_H, k_grid_500_H = VFI_HPI_grid_mat(500,500)

#plot(k_grid_50_H, V_50_H,
#label = "", legend = :bottomright, widths = [10], title = "Value Function")

Euler_500_H = VFI_Analytical_Results(500,G_kp_500_H)

#Euler graph HPI
plot(k_grid_500_H, Euler_500_H,linetype=:scatter,marker=(:diamond,3),
label = "HPI Euler Error", title= "HPI Euler Equation Error(%)", legend = :topright)
xlabel!("Capital")
ylabel!("Percentage Points")
#savefig("HW2_5b_Euler")
#Euler_200_H = VFI_Analytical_Results(200,G_kp_200_H)

#Value function graph HPI
plot(k_grid_500_H, V_500_H,
label = "n_k = 500", legend = :bottomright, widths = [10], title = " HPI Value Function")
xlabel!("Capital")
ylabel!("Value")
#savefig("HW2_5b_value")
#plot policy function HPI
data_H = zeros(500)
for i = 1:500
    data_H[i]= k_grid_500_H[G_kp_500_H[i]]
end
plot(k_grid_500_H, data_H,
label = "g(k)", legend = :bottomright, widths = [10], title = "HPI Policy Function")
plot!(k_grid_500_H,k_grid_500_H, legend = :bottomright,label="")
plot!([SS_values()[2]], seriestype="vline", label="Steady State")
xlabel!("Capital")
ylabel!("Capital")
#savefig("HW2_5b_policy")

println("----------------------------------------------------------------")
println("--MacQueen-Porteus Bounds--------------------------------------------------------------")

# Solve VFI with MPB
function Solve_VFI_MPB(n_k)
    # Get Grid
    k_grid = Make_K_Grid(n_k)
    # Utility matrix
    U_mat = [utility(k_grid[i],k_grid[j], opt_l(k_grid[i],k_grid[j])) for i in 1:n_k, j in 1:n_k]
    # Solve VFI
    V, G_kp, G_c = VFI_grid_MPB(x->T_grid_mat(x,U_mat,k_grid),k_grid)
    # Return Solution
    return V,G_kp, G_c, k_grid
end

# Fixed point with MPB
function VFI_grid_MPB(T::Function,k_grid)
    # Initialize variables for loop
    n_k    = length(k_grid) ; # Number of grid nodes
    V_old  = zeros(n_k)     ; # Initial value, a vector of zeros
    iter   = 0              ; # Iteration index
    V_dist = 1              ; # Initialize distance
    println(" ")
    println("------------------------")
    println("VFI - Grid Search - MPB - n_k=$n_k")
    for iter=1:max_iter
        # Update value function
        V_new, G_kp, G_c = T(V_old)
        # MPB and Distance
        MPB_l  = β/(1-β)*minimum(V_new-V_old)
        MPB_h  = β/(1-β)*maximum(V_new-V_old)
        V_dist = MPB_h - MPB_l
        # Update old function
        V_old  = V_new
        # Report progress
        if mod(iter,100)==0
            println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        end
        # Check Convergence
        if (V_dist<=dist_tol)
            # Recover value and policy functions
            V = V_old .+ (MPB_l+MPB_h)/2
            # Return
            println("VFI - Grid Search - MPB - n_k=$n_k")
            println("Iterations = $iter and Distance = ",100*V_dist)
            println("------------------------")
            println(" ")
            return V, G_kp, G_c
        end
    end
    # Report error for non-convergence
    error("Error in VFI - Grid Search - MPB - Solution not found")
end

# Define function for Value update and policy functions
function T_grid_mat(V_old,U_mat,k_grid)
    n_k    = length(V_old)
    V,G_kp = findmax( U_mat .+ β*repeat(V_old',n_k,1) , dims=2 )
    G_kp   = [G_kp[i][2] for i in 1:n_k] # G_kp is a Cartesian index
    opl = zeros(n_k)
    for i=1:n_k
        opl[i] = opt_l(k_grid[i],k_grid[G_kp[i]])
    end
    G_c    = (z*(k_grid.^α).*opl.^(1-α)) .- k_grid[G_kp]
    return V, G_kp, G_c
end

 @time V_600_MPB, G_kp_600_MPB, G_c_600_MPB, k_grid_600_MPB = Solve_VFI_MPB(600)

 plot(k_grid_600_MPB, V_600_MPB,
 label = "n_k = 600", legend = :bottomright, widths = [10], title = " MPB Value Function")
 xlabel!("Capital")
 ylabel!("Value")
 #savefig("HW2_5c_Value")

 Euler_600_MPB = VFI_Analytical_Results(600,G_kp_600_MPB)
 plot(k_grid_600_MPB, Euler_600_MPB,linetype=:scatter,marker=(:diamond,3),
 label = "MPB Euler Error", title= "MPB Euler Equation Error(%)", legend = :topright)
 xlabel!("Capital")
 ylabel!("Percentage Points")
 #savefig("HW2_5c_Euler")

 data_MPB = zeros(600)
 for i = 1:600
     data_MPB[i]= k_grid_600_MPB[G_kp_600_MPB[i]]
 end
 plot(k_grid_600_MPB, data_MPB,
 label = "g(k)", legend = :bottomright, widths = [10], title = "MPB Policy Function")
 plot!(k_grid_600_MPB,k_grid_600_MPB, legend = :bottomright,label="")
 plot!([SS_values()[2]], seriestype="vline", label="Steady State")
 xlabel!("Capital")
 ylabel!("Capital")
#savefig("HW2_5c_policy")

#6a path approximation
ss_mod = SS_values()[2]*0.8

function kp_get(k, grid, policy)
    path_a = zeros(20)
    path_a[1]= k
    for i = 1:length(path_a)
        for j = 1:length(grid)
            if path_a[i] >grid[j]
                println(path_a[i], " > ", grid[j])
            else
                println("update, go to ", policy[j-1])
                path_a[i+1] = policy[j]
                break
            end

        end
        if i == length(path_a)-1
            break
        end
    end
    return path_a
end

path_k = kp_get(ss_mod, k_grid_600_MPB,  data_MPB)
run = 19
global data_path = zeros(run+1, 6)
global data_path[:,1] = path_k

for  i in 1:run
    #l
    global data_path[i,6] = opt_l(data_path[i,1],data_path[i+1,1])
    #r
    global data_path[i,2] = z*α*data_path[i,1]^(α-1)*data_path[i,6]^(1-α)
    #w
    global data_path[i,3] = z*(1-α)*data_path[i,1]^(α)*data_path[i,6]^(-α)
    #y
    global data_path[i,4] = z*data_path[i,1]^(α)*data_path[i,6]^(1-α)
    #global k = k_prime(k)
    #capital
    #global data[i+1,1] = k
    #c
    global data_path[i,5] = data_path[i,4]-data_path[i+1,1]
    println(data_path[i,1])
end

ss_data_a = [ones(run+1,1)*SS_values()[2] ones(run+1,1)*SS_values()[5] (
ones(run+1,1)*SS_values()[6]) ones(run+1,1)*SS_values()[4] (
ones(run+1,1)*SS_values()[3]) ones(run+1,1)*SS_values()[1]]

x_axis = 1:10
println("--------------------------")
plot(x_axis, data_path[1:10,1:6], layout = (3, 2),
label = ["k" "r" "w" "y" "c" "l"], legend = :bottomright)

plot!(x_axis, ss_data_a[1:10,1:6], layout = (3, 2), ylims = [() () () () () (0.35,0.45)],
label = ["ss k" "ss r" "ss w" "ss y" "ss c" "ss l"], linestyle = :dot)
xlabel!("Iteration")
#savefig("HW2_6a_k")

#6b
orig_ss_k = SS_values()[2]

z = 1.05
ss_data_b = [ones(run+1,1)*SS_values()[2] ones(run+1,1)*SS_values()[5] (
ones(run+1,1)*SS_values()[6]) ones(run+1,1)*SS_values()[4] (
ones(run+1,1)*SS_values()[3]) ones(run+1,1)*SS_values()[1]]

@time V_600_MPB_b, G_kp_600_MPB_b, G_c_600_MPB_b, k_grid_600_MPB_b = Solve_VFI_MPB(600)

data_MPB_b = zeros(600)
for i = 1:600
    data_MPB_b[i]= k_grid_600_MPB_b[G_kp_600_MPB_b[i]]
end

path_k_b = kp_get(orig_ss_k, k_grid_600_MPB_b,  data_MPB_b)
global data_path_b = zeros(run+1, 6)
global data_path_b[:,1] = path_k_b

for  i in 1:run
    #l
    global data_path_b[i,6] = opt_l(data_path_b[i,1],data_path_b[i+1,1])
    #r
    global data_path_b[i,2] = z*α*data_path_b[i,1]^(α-1)*data_path_b[i,6]^(1-α)
    #w
    global data_path_b[i,3] = z*(1-α)*data_path_b[i,1]^(α)*data_path_b[i,6]^(-α)
    #y
    global data_path_b[i,4] = z*data_path_b[i,1]^(α)*data_path_b[i,6]^(1-α)
    #global k = k_prime(k)
    #capital
    #global data[i+1,1] = k
    #c
    global data_path_b[i,5] = data_path_b[i,4]-data_path_b[i+1,1]
    println(data_path_b[i,1])
end

plot(x_axis, data_path_b[1:10,1:6], layout = (3, 2),
label = ["k" "r" "w" "y" "c" "l"], legend = :bottomright)

plot!(x_axis, ss_data_b[1:10,1:6], layout = (3, 2), ylims = [(0.07, 0.085) (1.0, 1.06) () (0.23, 0.25) (0.16, 0.17) (0.35,0.45)],
label = ["ss k" "ss r" "ss w" "ss y" "ss c" "ss l"], linestyle = :dot)
xlabel!("Iteration")
#savefig("HW2_6a_k")

#=
@time V_100_b, G_kp_100_b, G_c_100_b, k_grid_100_b = VFI_grid_loop(100)

data_b = zeros(100, 1)

for i = 1:100
    data_b[i]= k_grid_100_b[G_kp_100_b[i]]
end
path_k_VFI = kp_get(orig_ss_k, k_grid_100_b, data_b)

global data_path_VFI = zeros(run+1, 6)
global data_path_VFI[:,1] = path_k_VFI

for  i in 1:run
    #l
    global data_path_VFI[i,6] = opt_l(data_path_VFI[i,1],data_path_VFI[i+1,1])
    #r
    global data_path_VFI[i,2] = z*α*data_path_VFI[i,1]^(α-1)*data_path_VFI[i,6]^(1-α)
    #w
    global data_path_VFI[i,3] = z*(1-α)*data_path_VFI[i,1]^(α)*data_path_VFI[i,6]^(-α)
    #y
    global data_path_VFI[i,4] = z*data_path_VFI[i,1]^(α)*data_path_VFI[i,6]^(1-α)
    #global k = k_prime(k)
    #capital
    #global data[i+1,1] = k
    #c
    global data_path_VFI[i,5] = data_path_VFI[i,4]-data_path_VFI[i+1,1]
    #println(data_path_b[i,1])
end

plot(x_axis, data_path_VFI[1:10,1:6], layout = (3, 2),
label = ["k" "r" "w" "y" "c" "l"], legend = :bottomright)

plot!(x_axis, ss_data_b[1:10,1:6], layout = (3, 2), ylims = [() () () () (0.14,0.16) (0.35,0.45)],
label = ["ss k" "ss r" "ss w" "ss y" "ss c" "ss l"], linestyle = :dot)
xlabel!("Iteration")

plot(k_grid_100_b, data_b,
label = "g(k)", legend = :bottomright, widths = [10], title = "MPB Policy Function")
plot!(k_grid_100_b,k_grid_100_b, legend = :bottomright,label="")
plot!([SS_values()[2]], seriestype="vline", label="Steady State")
xlabel!("Capital")
ylabel!("Capital")

=#
