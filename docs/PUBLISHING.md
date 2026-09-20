# GitHub publication checklist

**Public release date:** 2026-09-20

**Repository:** https://github.com/r-alia-sicily/cassandra-r3v

Use this checklist after the validated source archive has been unpacked into the new repository.

## Repository

- [x] Create the public repository `r-alia-sicily/cassandra-r3v`.
- [ ] Put the contents of this directory at the repository root.
- [ ] Confirm that no NASA data, workspace, trained model, evaluation, credential, or personal path is staged.
- [ ] Run the automated suite and source compilation.
- [ ] Run Evaluation and Full Test for all eight experiments; preserve every paired-comparison JSON.
- [ ] Review `README.md`, the PDF/LaTeX user manual, `LICENSE`, `CITATION.cff`, `SECURITY.md`, and `CONTRIBUTING.md`.
- [ ] Push the default branch and wait for GitHub Actions to pass.

Suggested local sequence:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Release Cassandra R3v 1.0.0"
git remote add origin https://github.com/r-alia-sicily/cassandra-r3v.git
git push -u origin main
```

## Immutable release

- [ ] Create the annotated tag only from the accepted commit.
- [ ] Push the tag.
- [ ] Create a GitHub release titled `Cassandra R3v 1.0.0`.
- [ ] Attach the validated wheel, source release ZIP, and `SHA256SUMS.txt`.
- [ ] Paste the release notes from `docs/RELEASE_NOTES.md`.
- [ ] Verify every asset by downloading it from GitHub and comparing its SHA-256 digest.

```bash
git tag -a v1.0.0 -m "Cassandra R3v 1.0.0"
git push origin v1.0.0
```

Do not move or recreate the tag after publication. If a defect is found, publish a new semantic version and describe the change.

## Archival record

- [ ] Connect the repository to an archival service such as Zenodo.
- [ ] Archive the GitHub release.
- [ ] Add the DOI to the GitHub release description and repository metadata.
- [ ] Update the article's software-availability statement with repository URL, tag, commit SHA, DOI, and GPL-3.0-or-later license.
- [ ] If citation metadata need the DOI, update them in a later commit; do not rewrite the accepted tag.

## Article gate

The article may use Cassandra results only after the canonical campaign in `docs/REPRODUCIBILITY.md` has completed and its evidence bundle has been checked. Claims about AutoStructure must include the paired uncertainty result and must not turn an `inconclusive` interval into a demonstrated improvement. The bundled Full Test report validates the release software; it does not substitute for the article campaign.
