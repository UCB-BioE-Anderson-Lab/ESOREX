# Demonstrations

Selected case studies showing different aspects of ESOREX behavior. Each page walks through one enzyme system end-to-end: the enzyme, the data, how the model represents its substrates, what it learns about the active site, and where it fails. Together they show the model's two regimes, regioselectivity (picking one site on a scaffold) and substrate specificity (predicting rates across a family).

---

## [TyrB, reading an aminotransferase as free energy](demonstrations/tyrb_extrapolation.md)

*E. coli* TyrB is a PLP-dependent aminotransferase with a strong preference for aromatic amino acid side chains. [Onuffer & Kirsch](https://pubmed.ncbi.nlm.nih.gov/8528072/) measured its activity against 23 substrates spanning five orders of magnitude under identical conditions, an unusually complete quantitative dataset.

Training on just the 9 natural amino acids, the energetic model recovers what the active site rewards (a large aromatic side chain), exactly interpolates the training measurements, and predicts 14 unnatural analogs, most within a few-fold (held-out Spearman ρ ≈ 0.73; median |log₁₀ error| ≈ 0.60, i.e. about 4-fold). It is explicit about its clearest held-out failure, a long aliphatic side chain (2-aminooctanoate) it under-predicts, flagging it as extrapolation rather than trusting it; a second experiment feeds that outlier back into training and shows the model absorbs it exactly without eroding the rest.

---

## [FucTIII, regioselectivity from inferred negatives](demonstrations/fuctiii_regioselectivity.md)

*Helicobacter pylori* FucTIII is an α1,3/4-fucosyltransferase that places its fucose on a **preferred** hydroxyl among the many on an oligosaccharide (other sites may react more slowly, so this is a preferred-site prediction, not a permissible/impossible one), distinguishing chemically similar sites on one scaffold, trained on just two reactions expanded by the **inferred negative** (every hydroxyl the enzyme could have used and passed over). The reactive site becomes a low-energy positive and each passed-over site an assigned high-energy negative (see the vignette for how that energy is set, no rate is measured there); from those two reactions the model picks the correct hydroxyl on 6 of 10 held-out compounds.

---

← [Model: energetic specificity](model.md)

*ESOREX docs · [Home](index.md) · [Theory](theory.md) · [Representation](representation.md) · [Model](model.md) · **Demonstrations***
