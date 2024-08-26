using Documenter, DocumenterCitations, BcdiSimulate

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiStrain.jl",
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
    repo = "github.com/byu-cig/BcdiStrain.jl.git",
)
