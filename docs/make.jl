using Documenter, DocumenterCitations, BcdiSimulate

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiSimulate.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiSimulate"=>"index.md",
        "Usage"=>[
            "Atomic"=>"usage/atom.md"
        ]
    ],
    plugins = [bib]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiSimulate.jl.git",
)
