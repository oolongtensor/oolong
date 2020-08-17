using TensorDSL

V4 = VectorSpace(4)
V67 = VectorSpace(67)
V3 = VectorSpace(3)
V1000 = VectorSpace(1000)

A = VariableTensor("A", V4, V4, V3, V67)
B = VariableTensor("B", V4, V4, V3, V67)
C = VariableTensor("C", V67', V3')

a = FreeIndex(V4, "a")
b = FreeIndex(V4, "b")
x = FreeIndex(V67, "x")
y = FreeIndex(V3, "y")
z = FreeIndex(V67, "z")

expr = componenttensor((A + B)[a, b, y, x] ⊗ C[x', y'], a, b) + componenttensor(B[a, b, 3, 1], a, b)
expr = (expr - expr)⊗A + (3*expr)⊗B
expr = +([expr[1,2,3,4,1,i] for i in 1:1000])
@time assign(expr, B => Tensor(fill(2.4, (4,4,3,67)), V4, V4, V3, V67))
println(expr.shape)
#  13.865809 seconds (15.79 M allocations: 790.048 MiB, 1.88% gc time) for Robin dict
# 13.006852 seconds (10.74 M allocations: 544.041 MiB, 1.52% gc time) for dict
# 9.675559 seconds (14.79 M allocations: 696.251 MiB, 4.18% gc time) for none