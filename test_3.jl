using Convex, SCS

# 定义问题
x = [1, 2]
y = [3, 4]
z = [5, 6]
a = Variable()
b = Variable()
c = Variable()
w = a * x + b * y + c * z
problem = minimize(sum(w), [0 <= a, a <= 1, 0 <= b, b <= 1, 0 <= c, c <= 1, a + b + c == 1])

# 求解问题
solve!(problem, SCS.Optimizer)

# 获取最优解
optimal_a = evaluate(a)
optimal_b = evaluate(b)
optimal_c = evaluate(c)
optimal_w = evaluate(w)

println(optimal_a)
println(optimal_b)
println(optimal_c)
println(optimal_w)
