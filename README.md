# NLP_CLASS

Course project: reproduce and extend OWL, "Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity" (Yin et al., ICML 2024, [arXiv 2310.05175](https://arxiv.org/abs/2310.05175), [code](https://github.com/luuyin/OWL)).

Authors: Tuan Khang Phan, Xuan Gia Han Nguyen (University of South Florida).

## Files
- [proposal/proposal.pdf](proposal/proposal.pdf) - project proposal (ACL template, 2 pages + references)
- [proposal/proposal.tex](proposal/proposal.tex), [proposal/references.bib](proposal/references.bib) - LaTeX source and BibTeX
- [docs/gpu_resources.md](docs/gpu_resources.md) - where to get GPUs (USF GAIVI, Kaggle, Modal, paid options)
- [2310.05175v4.pdf](2310.05175v4.pdf) - the paper

## Build the proposal
```
cd proposal
latexmk -pdf proposal.tex
```
On Overleaf, upload the `proposal/` folder and compile with pdfLaTeX.

## Plan in short
- Reproduce Table 3, Table 4, Table 8, Figure 1 and Figure 2 on LLaMA-7B (plus OPT-6.7B and LLaMA-13B).
- Extension: test OWL on Qwen2.5 (0.5B to 7B) and Llama-3.x (1B to 8B), check whether the gain still grows as models get smaller and whether the LOD shape predicts the gain.
