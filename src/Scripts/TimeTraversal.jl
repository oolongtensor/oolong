using TensorDSL

V4 = VectorSpace(4)
V67 = VectorSpace(67)
V3 = VectorSpace(3)

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
expr = ((expr[1] + expr[2])⊗B)[1, 3]⊗componenttensor(C[x', y'], y', x')
expr = (expr + expr + expr) ⊗ expr[1, 2]
expr = expr[a, 1, 2, 3, 4, 2, 4, 2, 1]⊗expr[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
expr =componenttensor(expr[x, 1,  2, 1,2, 1, 2, z'], a, x, z')
expr = (expr ⊗ expr + expr ⊗ expr)[1,2,3, 1, 2, x']⊗B[1, 2, 3, x]
@time assign(expr, B => Tensor(fill(2.4, (4,4,3,67)), V4, V4, V3, V67))
println(expr.shape)
# 5.550246 seconds (6.59 M allocations: 342.755 MiB, 2.12% gc time) without dictionary
# 8.025058 seconds (10.00 M allocations: 506.032 MiB, 2.33% gc time) with normal dictionary
# 10.860308 seconds (14.69 M allocations: 734.415 MiB, 2.36% gc time) with Robin dictionary