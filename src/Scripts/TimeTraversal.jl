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

expr = componenttensor((A + B)[a, b, y, x] ⊗ C[x', y'], a, b) + componenttensor(B[a, b, 3, 1], a, b)
expr = (expr - expr)⊗A
@time assign(expr, B => Tensor(fill(2.4, (4,4,3,67)), V4, V4, V3, V67))
println("Cool")