using Printf
using Statistics
using Plots
run = 25
a = 1/3
b = 0.85
z = 1
function parameter(a,b,v)
    global k_s = (a*b*v)^(1/(1-a))
    global A = a/(1-a*b)
    global k_p = x -> (b*A*v*x^a)/(1+b*A)
    global c = (x,y) -> v*x^a-y
    global r = x -> a*v*x^(a-1)
    global w = x -> (1-a)*v*x^a
    global y = x -> v*x^a
end
parameter(a,b,z)
ss_k = k_s
k = k_s*0.8

k_1=zeros(run+1)
r_1=zeros(run+1)
w_1=zeros(run+1)
y_1=zeros(run+1)
c_1=zeros(run+1)

ss_k_1=ones(run+1)*k_s
ss_r_1=ones(run+1)*r(k_s)
ss_w_1=ones(run+1)*w(k_s)
ss_y_1=ones(run+1)*y(k_s)
ss_c_1=ones(run+1)*c(k_s, k_s)

k_1[1]=k

for  i in 1:run
    global r_1[i] = r(k)
    global w_1[i] = w(k)
    global y_1[i] = y(k)
    global k = k_p(k)
    global k_1[i+1] = k
    global c_1[i] = c(k_1[i],k_1[i+1])
    if i== run
        global r_1[i+1] =r(k)
        global w_1[i+1] =w(k)
        global y_1[i+1] =y(k)
    else
    end
end

x_axis = 1:10
println("--------------------------")
plot(x_axis, k_1[1:10],
label = ["k"])

plot!(x_axis, ss_k_1[1:10],
label = ["ss k"])
savefig("HW1_k_1")


x_axis = 1:10
println("--------------------------")
plot(x_axis, r_1[1:10],
label = ["r"])

plot!(x_axis, ss_r_1[1:10],
label = ["ss r"])
savefig("HW1_r_1")

x_axis = 1:10
println("--------------------------")
plot(x_axis, w_1[1:10],
label = ["w"])

plot!(x_axis, ss_w_1[1:10],
label = ["ss w"])
savefig("HW1_w_1")

x_axis = 1:10
println("--------------------------")
plot(x_axis, y_1[1:10],
label = ["y"])

plot!(x_axis, ss_y_1[1:10],
label = ["ss y"])
savefig("HW1_y_1")

x_axis = 1:10
println("--------------------------")
plot(x_axis, c_1[1:10],
label = ["c"])

plot!(x_axis, ss_c_1[1:10],
label = ["ss c"])
savefig("HW1_c_1")

parameter(a,b,1.05)
k_2=zeros(run+1)
r_2=zeros(run+1)
w_2=zeros(run+1)
y_2=zeros(run+1)
c_2=zeros(run+1)

ss_k_2=ones(run+1)*k_s
ss_r_2=ones(run+1)*r(k_s)
ss_w_2=ones(run+1)*w(k_s)
ss_y_2=ones(run+1)*y(k_s)
ss_c_2=ones(run+1)*c(k_s, k_s)

k_2[1] = ss_k

for  i in 1:run
    global r_2[i] = r(ss_k)
    global w_2[i] = w(ss_k)
    global y_2[i] = y(ss_k)
    global ss_k = k_p(ss_k)
    global k_2[i+1] = ss_k
    global c_2[i] = c(k_2[i],k_2[i+1])
    if i== run
        global r_2[i+1] =r(ss_k)
        global w_2[i+1] =w(ss_k)
        global y_2[i+1] =y(ss_k)
    else
    end
end
x_axis = 1:10
println("--------------------------")
plot(x_axis, k_2[1:10],
label = ["k"])

plot!(x_axis, ss_k_2[1:10],
label = ["ss k"])
xlabel!("Iteration")
savefig("HW1-k-2")

x_axis = 1:10
println("--------------------------")
plot(x_axis, r_2[1:10],
label = ["w"])

plot!(x_axis, ss_r_2[1:10],
label = ["ss r"])
xlabel!("Iteration")
savefig("HW1-r-2")

x_axis = 1:10
println("--------------------------")
plot(x_axis, w_2[1:10],
label = ["w"])

plot!(x_axis, ss_w_2[1:10],
label = ["ss w"])
xlabel!("Iteration")
savefig("HW1-w-2")

x_axis = 1:10
println("--------------------------")
plot(x_axis, y_2[1:10],
label = ["y"])

plot!(x_axis, ss_y_2[1:10],
label = ["ss y"])
xlabel!("Iteration")
savefig("HW1-y-2")

x_axis = 1:10
println("--------------------------")
plot(x_axis, c_2[1:10],
label = ["c"])

plot!(x_axis, ss_c_2[1:10],
label = ["ss c"])
xlabel!("Iteration")
savefig("HW1-c-2")
