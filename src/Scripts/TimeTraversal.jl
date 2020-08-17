using TensorDSL

V4 = VectorSpace(4)
V67 = VectorSpace(67)
V3 = VectorSpace(3)
V1000 = VectorSpace(1000)

A = VariableTensor("A", V4, V4, V3, V67)
B = VariableTensor("B", V4, V4, V3, V67)
C = VariableTensor("C", V67', V3')
D = VariableTensor("D", V1000)

a = FreeIndex(V4, "a")
b = FreeIndex(V4, "b")
x = FreeIndex(V67, "x")
y = FreeIndex(V3, "y")
z = FreeIndex(V67, "z")

expr = componenttensor((A + B)[a, b, y, x] ⊗ C[x', y'], a, b) + componenttensor(B[a, b, 3, 1], a, b)
expr = ((expr - expr)⊗A + (3*expr)⊗B) ⊗D
expr = +([expr[1,2,3,4,1,1,i] for i in 1:1000]...)
@time assign(expr, B => Tensor(fill(2.4, (4,4,3,67)), V4, V4, V3, V67))
println(expr.shape)
# 14.922725 seconds (11.16 M allocations: 557.825 MiB, 2.01% gc time)
# 6.296863 seconds (7.72 M allocations: 390.513 MiB, 2.03% gc time)for dict
#  5.883682 seconds (6.25 M allocations: 311.498 MiB, 2.53% gc time)