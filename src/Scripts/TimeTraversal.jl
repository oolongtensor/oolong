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
expr = ((expr - expr)⊗A + (3*expr)⊗B) ⊗ D
expr = +([expr[1,2,3,4,1,1,i] for i in 1:50]...)
expr = +([(D⊗expr)[i] for i in 1:1000]...) 
@time assign(expr, B => Tensor(fill(2.4, (4,4,3,67)), V4, V4, V3, V67))
println(expr.shape)
# 28.266366 seconds (11.97 M allocations: 598.007 MiB, 1.44% gc time) for Robin dict
# 27.349857 seconds (8.26 M allocations: 417.624 MiB, 1.05% gc time) for dict
# 84.748977 seconds (80.31 M allocations: 3.421 GiB, 13.89% gc time) for nothing