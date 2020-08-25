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
a = ScalarVariable("a")
aTensor = Tensor(a)
arrayG = [cos(a), 4*sin(a), 4, 7im]
G = Tensor(arrayG, RnSpace(4))

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
        @test Base.cos(ScalarVariable("x")) isa CosineOperation
        @test Base.tan(B[1, 2]) isa TangentOperation
        @test_throws MethodError Base.sin(A)
        @test_throws MethodError Base.cos(A)
        @test_throws MethodError Base.tan(A)
    end
    @testset "Differentation" begin
        @test Base.diff(aTensor, a) == ConstantTensor(1)
        @test Base.diff(aTensor, ScalarVariable("z")) == ZeroTensor()
        @test Base.diff(aTensor + ZeroTensor(), a) == ConstantTensor(1)
        @test Base.diff(Base.sin(aTensor), a) == Base.cos(aTensor)
        @test Base.diff(Base.cos(aTensor), a) == -Base.sin(aTensor)
        @test Base.diff(Base.sin(a * ScalarVariable("z")), a) == Base.cos(aTensor * ScalarVariable("z")) * ScalarVariable("z")
    end
    @testset "TreeVisitor" begin
        @testset "Update children" begin
            @test updatechildren(A + B, A, B, B).children == (A, B, B)
            @test updatechildren(A⊗B, A, E).children == (A, E)
            @test updatechildren(A[x, 1], A, fixedindices).children == (A, fixedindices,)
            @test updatechildren(A[x, y]*D[y', z], (B[x, y]*D[y', z]).children...) == B[x, y]*D[y', z]
            @test updatechildren(sin(ScalarVariable("z")), Tensor(2)) == sin(Tensor(2))
            @test updatechildren(cos(ScalarVariable("z")), Tensor(2)) == cos(Tensor(2))
            @test updatechildren(tan(ScalarVariable("z")), Tensor(2)) == tan(Tensor(2))
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
            @test assign(aTensor, a=>4) == ConstantTensor(4)
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
            @test togem(D) == gem.Variable("D", (2, nothing))
            @test togem(Z) == gem.Zero((3, 2))
        end
    end
end
