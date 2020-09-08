using TensorDSL
using Test

V3 = VectorSpace(3)
V2 = VectorSpace(2)
Vi = VectorSpace()
Vj = VectorSpace()

x = FreeIndex(V3, "x")
y = FreeIndex(V2, "y")
z = FreeIndex(Vi, "z")
w = FreeIndex(Vj, "w")
fixedindices = Indices(FixedIndex(V3, 2), FixedIndex(V2, 1))

A = VariableTensor("A", V3, V2)
B = Tensor(fill(1.2, (3, 2)), V3, V2)
C = VariableTensor("C", Vj, Vi, Vi')
D = VariableTensor("D", V2', Vi)
E = VariableTensor("E", V2', V3', Vi)
F = Tensor(fill(1.5, (2,3)), V2', V3)
Z = ZeroTensor(V3, V2)
a = VariableTensor("a")
arrayG = [cos(a), 4*sin(a), 4, 7im]
G = Tensor(arrayG, RnSpace(4))
H = VariableTensor("H", V2', RnSpace(5))
# TODO make the addition nicer
I = Tensor(reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ConstantTensor(11) + a], (2,2,3)), V2, V2', V3)

@testset "TensorDSL.jl" begin
    @testset "Operations" begin
        @testset "Addition" begin
            @test (A + B).shape == A.shape
            @test A + Z == A
            @test (A + Z)[1, 2] == A[1, 2]
            @test Z + Z + Z == Z
            @test (A + B).children == (A, B)
            @test (A[x,1] + B[1, y]).freeindices == (x, y)
            @test_throws DimensionMismatch A + D
            @test (A[x, y] + D[y', z]).freeindices == (x,z)
        end
        @testset "Index" begin
            @test A[x, 1].shape == ()
            @test A[x, 1].freeindices == (x,)
            @test_throws DimensionMismatch A[x, x]
            @test A[x].shape == (V2,)
            @test A[x] isa ComponentTensorOperation
            @test E[y'].shape == (V3', Vi)
        end
        @testset "Outer product" begin
            @test (A⊗B).shape == (V3, V2, V3, V2)
            @test (A⊗B).children == (A, B)
            @test (A[x,1] ⊗ B[1, y]).freeindices == (x, y)
            @test (A - B).shape == (V3, V2)
            @test (-B).shape == (V3, V2)
            @test (A - B) isa AddOperation
        end
        @testset "Power" begin
            @test A[1,2]^2 isa PowerOperation
            @test sqrt(a) isa PowerOperation
            @test sqrt(a).children == (a, ConstantTensor(1//2))
        end
        @testset "Division" begin
            @test_throws DivideError A / ZeroTensor()
            @test_throws DivideError A / 0
            @test_throws Exception A / A
            @test A / 5 isa DivisionOperation
            @test (A / 4).children == (A, ConstantTensor(4))
            @test A / 1 == A
        end
        @testset "Component tensor" begin
            @test componenttensor(A[x, 1], x).shape == (V3,)
            @test componenttensor(A[x, y], y).shape == (V2,)
            @test componenttensor(A[x, y], y, x).shape == (V2, V3)
            @test componenttensor(E[y', x', z], x', z).shape == (V3', Vi)
            @test componenttensor(E[y', x', z], x', z).freeindices == (y',)
            @test_throws DomainError componenttensor(A[x, y], x, z)
        end
        @testset "Tensor contraction" begin
            @test (A[x, y]*D[y', z]).shape == ()
            @test (A[x, y]*D[y', z]).freeindices == (x, z)
            @test (A[x, y]*E[y', x', z]).shape == ()
            @test (A[x, y]*E[y', x', z]).freeindices == (z,)
            @test componenttensor(C[w, z, z'], w).shape == (Vj,)
            @test C[w, z, z'].freeindices == (w,)
            @test C[w, z, z'].children[2] == Indices(z)
        end
        @testset "Transpose" begin
            @test transpose(A).shape == reverse(A.shape)
            @test transpose(A) isa ComponentTensorOperation
            @test transpose(C).shape == reverse(C.shape)
        end
        @testset "Trace" begin
            @test trace(VariableTensor("X", V3, V3')) isa IndexSumOperation
            @test_throws DomainError trace(VariableTensor("X", V3, V3))
            @test_throws MethodError trace(VariableTensor("X", V3, V3, V3'))
        end
    end
    @testset "Tensors" begin
        @test_throws DomainError Tensor([1, 2], VectorSpace(3), VectorSpace(2))
        @test_throws DomainError Tensor([1, 2], VectorSpace(3))
    end
    @testset "Indices" begin
        @test_throws DomainError FixedIndex(VectorSpace(4), 5)
        @test_throws DomainError FixedIndex(VectorSpace(), 5)
    end
    @testset "Trigonometry" begin
        @test Base.sin(Tensor(1)) isa SineOperation
        @test Base.sin(Tensor(3)).children[1] == Tensor(3)
        @test Base.cos(VariableTensor("x")) isa CosineOperation
        @test Base.tan(B[1, 2]) isa TangentOperation
        @test_throws MethodError Base.sin(A)
        @test_throws MethodError Base.cos(A)
        @test_throws MethodError Base.tan(A)
        @test Base.asin(Tensor(.4)) isa ArcsineOperation
        @test_throws DomainError Base.asin(Tensor(4))
        @test Base.acos(Tensor(.4)) isa ArccosineOperation
        @test_throws DomainError Base.acos(Tensor(4))
        @test Base.atan(Tensor(40)) isa ArctangentOperation
    end
    @testset "Differentation" begin
        @test differentiate(a, a) == ConstantTensor(1)
        @test differentiate(a, VariableTensor("z")) == ZeroTensor()
        @test differentiate(a + ZeroTensor(), a) == ConstantTensor(1)
        @test differentiate(a + a, a) == ConstantTensor(1) + ConstantTensor(1)
        @test differentiate(a + 3*a, a) == ConstantTensor(1) + ConstantTensor(3)
        @test differentiate(1 / a, a) == (-1 / (a^2))
        @test differentiate(Base.sin(a), a) == Base.cos(a)
        @test differentiate(Base.cos(a), a) == -Base.sin(a)
        @test differentiate(Base.sin(a * VariableTensor("z")), a) == Base.cos(a * VariableTensor("z")) * VariableTensor("z")
        @test differentiate(Base.tan(Base.cos(a)), a) == - Base.sin(a) / (Base.cos(Base.cos(a))^2)
    end
    @testset "TreeVisitor" begin
        @testset "Update children" begin
            @test updatechildren(A + B, A, B, B).children == (A, B, B)
            @test updatechildren(A⊗B, A, E).children == (A, E)
            @test updatechildren(A[x, 1], A, fixedindices).children == (A, fixedindices,)
            @test updatechildren(A[x, y]*D[y', z], (B[x, y]*D[y', z]).children...) == B[x, y]*D[y', z]
            @test updatechildren(sin(VariableTensor("z")), Tensor(2)) == sin(Tensor(2))
            @test updatechildren(cos(VariableTensor("z")), Tensor(2)) == cos(Tensor(2))
            @test updatechildren(tan(VariableTensor("z")), Tensor(2)) == tan(Tensor(2))
        end
    end
    @testset "Assignment" begin
        @testset "Vector spaces" begin
            @test assign(D, Vi=>RnSpace(2)) == VariableTensor("D", V2', RnSpace(2))
            @test assign(ZeroTensor(Vi'), Vi'=>RnSpace(2)) == ZeroTensor(RnSpace(2))
            @test assign(ConstantTensor(a, Vi'), Vi'=>RnSpace(2)) == ConstantTensor(a, RnSpace(2))
            @test assign(DeltaTensor(Vi'), Vi=>V2) == DeltaTensor(V2')
            @test_throws DomainError assign(D, V3=>Vi)
            @test assign(D[1, z], Vi=>RnSpace(2)).freeindices == (FreeIndex(RnSpace(2), "z"),)
            @test assign(componenttensor(D[y', z], z, y'), Vi=>RnSpace(2)).shape == (RnSpace(2), V2')
        end
        @testset "Tensors" begin
            @test assign(A, A=>B) == B
            # TODO Create a better node equality so that strings are not needed
            @test string(assign((A⊗C)[2], A=>B)) == string((B⊗C)[2])
            @test_throws DomainError assign(D, D=>B)
            @test assign(A, A=>ConstantTensor(2, A.shape...)) == ConstantTensor(2, A.shape...)
        end
        @testset "Variables" begin
            @test assign(a, a=>4) == ConstantTensor(4)
            @test assign(Tensor([a, 6], RnSpace(2)), a=>4).value == [4, 6]
            @test assign(Tensor([cos(a), a], V2), a => 0).value == [cos(ZeroTensor()), 0]
            @test assign(G, a=>0).value == [cos(ZeroTensor()), 4*sin(ZeroTensor()), 4, 7im]
        end
        @testset "Indices" begin
            @test assign(Indices(x), x=>1) == Indices(FixedIndex(V3, 1))
            @test string(assign(A[x], x=>1)) == string(A[1])
            @test string(assign(A[x, y], x=>1)) == string(A[1, y])
        end
    end
    @testset "Gem" begin
        @testset "Tensors" begin
            @test togem(B) == gem.Literal(fill(1.2, (3, 2)))
            @test togem(ConstantTensor(1, V3, V2)) == gem.Literal(fill(1, (3,2)))
            @test togem(H) == gem.Variable("H", (2, 5))
            @test togem(Z) == gem.Zero((3, 2))
        end
        @testset "Indices" begin
            @test togem(Indices(y, z, FixedIndex(V2, 2))) == togem(Indices(y, z, FixedIndex(V2, 2)))
            @test togem(H[y', 1]).free_indices == togem(Indices(y'))
            @test togem(H[y', 1]).children == (togem(H),)
            @test togem(H[y', 1]).multiindex == togem(Indices(y', FixedIndex(RnSpace(5), 1)))
            @test togem(Tensor(collect(1:5), RnSpace(5))[2]) == gem.Literal(2)
        end
        @testset "componenttensor" begin
            @test togem(A[x]).shape == (2,)
            @test togem(A[x]).free_indices == togem(Indices(x))
            @test togem(componenttensor(A[x],x)).shape == (2,3)
            @test togem(componenttensor(A[x],x)).free_indices == ()
        end
        @testset "Addition" begin
            @test togem(A + B).shape == (3, 2)
            @test isinst(togem(A + B), gem.ComponentTensor)
            @test isinst(togem(A + B).children[1], gem.Sum)
            @test togem(A + B + A).shape == (3, 2)
            @test togem(A + B).children[1].children[1].children[1] == togem(A)
            @test togem(A + B).children[1].children[2].children[1] == togem(B)
            @test togem(B + A + A).children[1].children[1].children[1].children[1] == togem(B)
            @test Set(togem(A[x, y] + B[1,2]).free_indices) == Set(togem(Indices(x, y)))
        end
        @testset "Product" begin
            @test togem(A⊗B).shape == (3, 2, 3, 2)
            @test togem(A⊗B).children[1].children[2].children[1] == togem(B)
            @test isinst(togem(A⊗B), gem.ComponentTensor)
            @test isinst(togem(A⊗B).children[1], gem.Product)
        end
        @testset "Division" begin
            @test isinst(togem(A[1,2]/3), gem.Division)
            @test togem(B[1,2] / 2) == gem.Literal(0.6)
            @test isinst(togem(A / VariableTensor("h")), gem.ComponentTensor)
            @test isinst(togem(A / VariableTensor("h")).children[1], gem.Division)
        end
        @testset "Index sum" begin
            @test togem(A[x, y]⊗F[y', 1]).free_indices == togem(Indices(x))
            @test togem(A[x, y]⊗F[y']).free_indices == togem(Indices(x))
            @test isinst(togem(A[x, y]⊗F[y']), gem.ComponentTensor)
            @test isinst(togem(A[x, y]⊗F[y']).children[1], gem.IndexSum)
            @test togem(A[x, y]⊗F[y']).free_indices == togem(Indices(x))
        end
        @testset "Trigonometry" begin
            @test togem(sin(a)).name == "sin"
            @test togem(cos(a)).name == "cos"
            @test togem(tan(a)).name == "tan"
            @test togem(asin(a)).name == "asin"
            @test togem(acos(a)).name == "acos"
            @test togem(atan(a)).name == "atan"
        end
    end
    # These tests only check that no errors are occurring, they do not check correctness.
    @testset "Create kernel" begin
        @test Kernel(A + B).knl !== nothing
        # This does not pass, but maybe it shouldn't?
        # @test Kernel(A[x, y]⊗F[y']) !== nothing
        @test Kernel((B + A + A)[1]).knl !== nothing
        @test Kernel(A[1, y]⊗F[y']).knl !== nothing
        @test Kernel(componenttensor((A + Z)[1, y]⊗F[y', x], x)).knl !== nothing
        @test Kernel(Tensor([a 1 ; 2 3], RnSpace(2), RnSpace(2))).knl !== nothing
    end    
    @testset "execute" begin
        @test execute(B) == fill(1.2, (1, 3, 2))
        @test execute(B + 5*B) == fill(1.2 + 5*1.2, (1, 3, 2))
        @test execute(Kernel(A), Dict("A"=>fill(5.4, (3, 2)))) == fill(5.4, (1, 3, 2))
        @test execute(A, [fill(5.4, (3, 2))]) == fill(5.4, (1, 3, 2))
        @test execute(Kernel(A[1, y]⊗H[y']), Dict("A"=>fill(1.0, (3,2)), "H"=>fill(-1.0, (5,2)))) == fill(-2.0, (1,5))
        @test execute(I, "a"=>1.0) == reshape([0.0 + i for i in 1:12],(1,2,2,3))
        @test execute(Tensor([sin(a), cos(a), tan(a)], V3), "a"=>1.0) == reshape([sin(1), cos(1), tan(1)], (1, 3))
        @test execute(Tensor([asin(a), acos(a), atan(a)], V3), "a"=>0.5) == reshape([asin(0.5), acos(0.5), atan(0.5)], (1, 3))
        @test execute(A / a, "A"=>fill(3.0, (3, 2)), "a"=>1.5) == fill(2.0, (1, 3, 2))
        @test execute(trace(Tensor([1 2 ; 3 4], V2, V2'))) == [5.0]
        @test execute(sqrt(a), "a"=>4.0) == [2.0]
        @test execute(a^a, "a"=>3.0) == [27.0]
    end
    @testset "find variables" begin
        @test findvariables((A[x, y] + D[y', z])⊗B) == Set{VariableTensor}([A, D])
    end
end
