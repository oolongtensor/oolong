include("AST.jl")
include("Operations.jl")
include("Tensors.jl")


ast = AST()
add = Add()
one = Scalar(1)
two = Scalar(2)
three = Scalar(3)
addchild!(ast, add, 0)
addchild!(ast, one, 1)
addchild!(ast, two, 1)
three_num = addnode!(ast, three)
addroot!(ast, add, three_num, 1)
println(ast.ast)

for vertex in vertices(ast.ast)
    println(vertex)
    println(outneighbors(ast.ast, vertex))
end
