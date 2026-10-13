# Course Material (zipped)

`Course_Material.zip` is split into 95 MB parts to fit GitHub's 100 MB file limit.
It contains `Labs/` and the lecture PDFs from `RAG Program/Course Material/`.
Excluded: `.env` files, virtual environments, caches, `.git` folders and Chroma vector DBs (rebuildable).

## Reassemble

PowerShell / cmd:

    cmd /c copy /b Course_Material.zip.part* Course_Material.zip

Git Bash / macOS / Linux:

    cat Course_Material.zip.part* > Course_Material.zip

SHA-256 of the reassembled zip: `38cbc3647e4f6b408571d3322f8f4faabe03954f04b3071ba6e29e5b19b9adc9`
