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
            "Vector Spaces" => "vectorspaces.md"
            "Indices" => "Indices.md"
            "Tensors" => "tensors.md",
            "Operations" => "operations.md",
            "Execution and building" => "execute.md"
        ],
    ]
)