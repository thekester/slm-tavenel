# Candidate replacement corpora for wiki-v1

**Reviewed:** 2026-10-21

**Recommendation:** no reviewed corpus is approved for the next confirmatory run or open-weight
release. The prepared EN/FR `french_open_science` candidate from
[French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons) retains
per-document source and attribution metadata and filters to CC0, CC BY 4.0, and Licence Ouverte 2.0,
but its source audit still blocks training pending provenance and privacy review. See the
[preparation review](FSC-PILOT-2026-10-20.md). MOT v1.11 has a documented 59.7% missing-author
metadata rate in English; this is a limitation to disclose, not an automatic bar to corpus-level
research. Its potential use for training and its suitability for public-weight redistribution are
separate decisions; see the [MOT review](MOT-VOA-2026-10-21.md).
For broader English pretraining, [Common Pile v0.1](https://common-pile.org/) is a documented,
large-scale open-licensed corpus, but includes CC BY-SA material and needs a release-specific
license review. No training or active run was changed during this source search.

## Candidate comparison

| Candidate | Strengths | Limits for SLM-Tavenel |
|---|---|---|
| [French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons), `french_open_science` subset | Dataset card reports 15.69B words overall and retains per-document source, license, terms, DOI and language. Its `French Open Science` collection is described as mostly CC BY, CC BY-SA, CC0 and Licence Ouverte. Filtering to CC0, CC BY and Licence Ouverte gives a promising French research-text candidate with document-level attribution. | Do not use `hal_open_access`: its card explicitly says it contains copyrighted papers distributed by HAL under additional conditions. Do not rely on aggregate licensing or OpenAlex metadata alone. Verify each selected record and publisher terms; OCR quality and document duplication also need review. The card itself is a curator statement, not an independent rights audit. |
| [Common Pile v0.1](https://common-pile.org/) | NeurIPS 2025 paper documents 8 TB from 30 sources, built from public-domain/open-licensed texts for pretraining, with released processing code and training mixtures. The paper excludes NC and ND material and explains its source-level due-diligence process. Strong English-heavy broad-domain candidate. | CC BY-SA is considered open and is included; its implications for the intended model-weight release need documented review. The paper says open licenses do not necessarily mean rights holders specifically contemplated LLM training. Large and diverse, but the complete mix is not a matched French-English parallel-domain design. Start with an audited subset and pinned version, not the entire 8 TB. |
| French public-sector documents under [Licence Ouverte 2.0](https://www.data.gouv.fr/pages/legal/licences/etalab-2.0) | The official license permits free reuse, adaptation and commercial use worldwide, with attribution. It is compatible with CC BY and ODC-BY. A genuinely bilingual sample could be formed from separately verified public-sector sources. | This is a license, not a ready-made corpus. Each dataset must explicitly carry LO2 and be checked for third-party text and personal-data constraints. Government/admin language is a narrow domain and may not represent general French. |
| [PleIAs English-PD](https://huggingface.co/datasets/PleIAs/English-PD) + [French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books) | Same broad genre in both languages; public-domain curation; author/title/source metadata; reported scale of 78.6B English words and 16.4B French words, ample for the 491.52M-token 10M runs. English books were limited to publications before 1884 to reduce rights-verification risk. BnF says academic/scientific use is free with source attribution. | French-PD-Books says the texts may be used for training and republished; BnF separately requires authorization/fees for defined commercial uses of its reproductions, even if the underlying work is public domain. Resolve this for model-weight distribution before any commercial release. Both collections have OCR and historical-language limitations. |
| [Multilingual Open Text (MOT) v1.11](https://github.com/bltlab/mot/releases/tag/v1.11) | Peer-reviewed English/French corpus; collection described as CC BY 4.0; pinned release and archive hashes provide corpus-level traceability. An empty author field can be reported as supplied, without inventing attribution or automatically excluding the article from corpus-level research. | Audit found 191,736 of 321,258 English articles (59.7%) lack author metadata. Disclose this limitation; author-dependent analyses must account for it. Public-weight distribution is a separate rights review because VOA terms restrict third-party/AP material. Corpus is news-domain, French is smaller, and privacy/quality/contamination checks remain necessary; see [the audit](MOT-VOA-2026-10-21.md). |
| [Digital Corpus of the European Parliament (DCEP)](https://joint-research-centre.ec.europa.eu/language-technology-resources/dcep-digital-corpus-european-parliament_en) | Official bilingual/multilingual institutional text; 1.37B words across 23 languages, including English and French; document and sentence alignment support careful language-pair or temporal studies. | The official usage terms grant reuse only for research purposes and require source attribution. That is not sufficient for unrestricted open-weight distribution; request clarification before considering it for a release. |
| [Open Bilbio corpus](https://www.data.gouv.fr/datasets/open-bilbio-corpus-for-content-analysis) | Dataset record declares CC BY and describes full-text scientific publications in mathematics, computing and statistics. Potentially useful for narrow science-domain evaluation or training. | The record describes a 650,000-publication corpus but exposes only two downloadable archives (28.6 MB and 14.6 MB); inspect their contents, source-level permissions and actual language/volume before relying on the dataset-level license for model training or weights. |
| [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) as a whole | Very large and multilingual, with document-level source and license fields in the newer release. | It mixes collections and terms. A 2025 NeurIPS dataset paper notes that Common Corpus does not retain complete per-document licensing across all sources and that OpenAlex license metadata can be inaccurate. Do not use the aggregate blindly. |

## Recommended audit path

For a scientific comparison of training methods, use the exact same frozen corpus, tokenizer,
token budget and evaluation protocol for every model variant. A French science subset is a much
cleaner rights-traceability path than wiki-v1, and can also support a focused French-language
experiment. To compare language coverage, make a corresponding English science subset using
documents whose individual source license is CC0 or CC BY. If a suitably sized, auditable English
subset cannot be assembled, state that the experiment is French-science-domain rather than implying
balanced bilingual general-language coverage.

Do not automatically combine licenses into one blanket label. Keep source-level license and
attribution records in the manifest. CC BY-SA is open for reuse but has share-alike obligations for
adapted material; before distributing weights trained on it, get a documented interpretation for
the exact intended distribution. For the most conservative first release review, omit CC BY-SA and
use only CC0, CC BY, and Licence Ouverte 2.0 records with confirmed provenance.

The French-Science-Commons card explicitly says that its `hal_open_access` section includes
copyrighted documents and separate publisher conditions. Exclude that section. The recommendation
above is limited to `french_open_science` rows whose license and terms can be verified individually;
that filter is necessary but has not cleared the current review gates. For MOT, missing author
metadata should be described accurately in the experiment/data card; it is not, by itself, a
scientific-validity veto. Decide separately whether the corpus and source terms permit distributing
the resulting weights.

## Why the book collections remain a separate option

The two book sets offer a practical bilingual comparison from broadly similar long-form prose and
reported volumes far above the planned training budget. The English card reports 736,214 titles and
78,566,669,909 words; it says the selected works were published before 1884 and meet the curator's
public-domain criteria. The French card reports 289,000 books and 16,407,292,362 words, identified
using author-death dates and Gallica identifiers.

This is a dataset-curator assertion, not a completed SLM-Tavenel rights review. BnF's official
conditions state that academic/scientific use of its documents is free with attribution, while
defined commercial uses require authorization and a fee. The dataset card notes this condition.
Before calling the corpus approved for a specific release, the data review must:

1. Pin the exact Hub commit and hash every downloaded file.
2. Preserve each source ID, title, author, publication metadata, and original library record URL.
3. Verify the public-domain rationale and relevant source terms; record whether the intended run,
   paper, checkpoints, and weights are academic/noncommercial or fall within BnF's commercial-use
   conditions for French-PD-Books.
4. Exclude records with missing or conflicting rights metadata rather than infer a license from the
   aggregate collection name.
5. Inspect OCR quality and remove non-text artefacts; split and deduplicate by book ID.
6. Freeze a language-balanced sample and held-out BPB documents before tokenizer training, then
   repeat benchmark decontamination on that exact corpus.

If the BnF condition cannot be reconciled with intended public or commercial weight distribution,
French-PD-Books does not pass the project's release requirements. MOT has clear published
licensing for both languages, but is too French-sparse and too narrow to substitute for the current
pretraining corpus at the same token budget. Use MOT only for a separately labelled small-corpus
robustness study, or source French text from rights-holder-published open documents and audit it
individually.

## Sources consulted

- PleIAs, [English Public Domain Books dataset card](https://huggingface.co/datasets/PleIAs/English-PD)
- PleIAs, [French Public Domain Books dataset card](https://huggingface.co/datasets/PleIAs/French-PD-Books)
- PleIAs, [French-Science-Commons dataset card](https://huggingface.co/datasets/PleIAs/French-Science-Commons)
- NeurIPS 2025, [The Common Pile v0.1 paper](https://papers.neurips.cc/paper_files/paper/2025/file/52acc050138d6f40dad6f12f91a4ce22-Paper-Datasets_and_Benchmarks_Track.pdf) and [project site](https://common-pile.org/)
- Etalab, [official Licence Ouverte 2.0 terms](https://www.data.gouv.fr/pages/legal/licences/etalab-2.0)
- BnF, [official rules for commercial use of reproductions](https://www.bnf.fr/fr/faire-une-utilisation-commerciale-dune-reproduction) and [reuse of BnF data](https://www.bnf.fr/fr/conditions-de-reutilisations-des-donnees-de-la-bnf)
- Palen-Michel et al. (2022), [MOT Release 1 paper](https://aclanthology.org/2022.lrec-1.224/)
- NeurIPS (2025), [The Common Pile v0.1 paper](https://papers.neurips.cc/paper_files/paper/2025/file/52acc050138d6f40dad6f12f91a4ce22-Paper-Datasets_and_Benchmarks_Track.pdf), discussion of license due diligence and Common Corpus metadata limitations
