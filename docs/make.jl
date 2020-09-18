using Documenter, TensorDSL

makedocs(sitename="Oolong",
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
    ),
    modules = [TensorDSL],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Abstract syntax tree" => "ast.md",
            "Vector Spaces" => "vectorspaces.md",
            "Indices" => "indices.md",
            "Tensors" => "tensors.md",
            "Operations" => "operations.md",
            "Execution and building" => "execute.md"
        ],
        "Improvements" => "future.md",
        "Python-Julia compatibility" => "python.md"
    ]
)

deploydocs(
    repo = "github.com/oolongtensor/oolong.git",
)