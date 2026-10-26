# Corpus de croissance bilingue, candidat v2

**Date :** 2026-10-25
**État :** candidat local reproductible, non validé juridiquement.
**EXP-0028 sur ce corpus :** entraînements 50M et 100M terminés; test interne final et scores de benchmarks externes encore à produire.

## Objectif et séparation des benchmarks

Cette version ajoute du texte web anglais et français à la sélection du [corpus candidat v1](RELIABLE-CORPUS-CANDIDATE-V1-2026-10-24.md), afin de disposer de données distinctes pour une prochaine expérience de croissance. Les benchmarks ARC, HellaSwag, Belebele et WinoGrande restent réservés à l’évaluation. Le constructeur ne copie aucune ligne de ces jeux dans le corpus. Il lit leurs questions et choix uniquement pour retirer les documents ayant un chevauchement exact normalisé de 13 tokens.

Les splits `dev` et `test` de ce corpus sont des partitions internes de documents HPLT, séparées par domaine source. Ils ne sont pas les benchmarks externes du projet.

## Nouvelle source Hugging Face

Le catalogue de découverte fourni est [Hugging Face Datasets, document/text, tri tendance](https://huggingface.co/datasets?modality=modality%3Adocument%2Cmodality%3Atext&sort=trending). La source retenue pour ce pilote est [ashtok897/european-hplt-v1](https://huggingface.co/datasets/ashtok897/european-hplt-v1/tree/51edb798468aa71b1000e950d3ab43bb435f72cd), au commit `51edb798468aa71b1000e950d3ab43bb435f72cd`.

La carte décrit 45 millions de documents pour 41 langues, environ 50,9 milliards de tokens estimés, un score qualité HPLT WDS d’au moins 10, les URL sources, la langue et des métadonnées de registre. Elle indique que CC0 est hérité de HPLT v3. Le corpus amont est issu de crawls web Common Crawl et Internet Archive. HPLT précise ne pas détenir les textes sources et laisse à l’utilisateur la responsabilité de vérifier le cadre juridique applicable. La déclaration CC0 du jeu dérivé est donc conservée comme information de licence déclarée, pas comme preuve indépendante des droits sur chaque page.

La carte indique aussi l’absence de filtrage PII au-delà des défauts HPLT. Les fichiers retenus ici ne fournissent pas un audit PII par document. Les filtres locaux ci-dessous réduisent quelques risques, sans fournir d’anonymisation ni de validation humaine.

## Portion traitée

Seuls les dix premiers fichiers du split upstream `train` ont été utilisés : `train-00000` à `train-00009-of-00181.parquet`, soit 2 500 000 lignes d’entrée et 7 958 157 907 octets. Chaque fichier source compte 250 000 lignes; le commit, le chemin, le nombre de lignes et le SHA-256 de chacun sont dans le manifeste local.

Le constructeur conserve les textes anglais et français avec une probabilité de langue d’au moins 0,90, les scores WDS d’au moins 10, les documents de 300 à 100 000 caractères et une proportion alphanumérique minimale de 0,48. Il écarte les classes de registre dominées par la traduction automatique (`MT`) et les paroles (`LY`), les formats non textuels non reconnus, les caractères de remplacement, certains identifiants directs, des signaux de contenu médical/sensible, les doublons exacts avec le corpus v1 et les chevauchements benchmark de 13 tokens. Chaque règle et chaque décision d’inclusion ou d’exclusion sont journalisées sans recopier les textes rejetés dans le journal.

Les URL sources, identifiants HPLT, scores de qualité, étiquettes de licence, révisions, splits et empreintes des textes retenus sont conservés avec les enregistrements. Les domaines eTLD+1, calculés avec la liste Public Suffix embarquée, sont assignés déterministement à `train` (90 %), `dev` (5 %) ou au holdout du corpus (5 %).

## Résultat

| Split | Documents v1 | HPLT retenus | Documents v2 | Tokens HPLT | Tokens totaux v2 |
|---|---:|---:|---:|---:|---:|
| Train | 61 435 | 211 171 | 272 606 | 334 015 243 | 375 290 224 |
| Dev | 8 718 | 10 225 | 18 943 | 14 646 319 | 18 855 038 |
| Holdout corpus | 7 461 | 11 121 | 18 582 | 16 077 665 | 21 332 741 |
| **Total** | **77 614** | **232 517** | **310 131** | **364 739 227** | **415 478 003** |

Les 334,0 millions de tokens HPLT de train dépassent d’environ 8,1 fois les 41,3 millions du train v1; le train v2 contient environ 9,1 fois autant de tokens que v1. La distribution des tokens de train v2 est **62,3 % anglais et 37,7 % français**. Ces comptes utilisent le tokenizer SLM-Tavenel v4, SHA-256 `93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb`; ils ne prédisent pas une amélioration de performance.

Sur les 2,5 millions de lignes examinées, 232 517 documents ont été retenus : 136 346 anglais et 96 171 français. Les principales exclusions sont 2 155 248 documents hors EN/FR, 45 858 sous le seuil de confiance de langue, 30 176 de registre `MT` ou `LY`, 22 588 avec un motif d’identifiant direct et 12 792 avec un signal de sujet sensible. Cinq documents ont été retirés pour chevauchement exact avec les benchmarks. Aucun benchmark n’est présent dans les sorties retenues.

## Reproduction et artefacts

Installer les dépendances de curation (`pip install -e ".[data-curation]"`), construire d’abord le v1, puis lancer :

```powershell
python scripts/build_reliable_corpus_candidate.py
python scripts/build_hplt_growth_candidate.py
```

Les sorties locales, ignorées par Git, sont dans `data/processed/slm-tavenel-growth-candidate-v2/`. Le manifeste enregistre les hashes de toutes les partitions et des shards, le hash du manifeste parent, celui de la carte Hugging Face épinglée, les entrées benchmark et le tokenizer. Les entrées HPLT sont dans `data/raw/european-hplt-v1-2026-10-25/`.

| Sortie | SHA-256 |
|---|---|
| Train | `b15d064e87ea0108261daf0e68c948c79fdfe7090840efc83b78f10353d058d3` |
| Dev | `227794ab099490405fd6fe039a028f35e7002c89e33ab19f5e6d0a6f547a8e98` |
| Holdout corpus | `92936fc3790d82d3da0dd5ee2450b037bc0c2b9d16785a8c78cdce07b65e9f15` |

Un audit local a confirmé les nombres de lignes, ces hashes, 310 131 identifiants uniques, 310 131 hashes de texte uniques et la présence de l’URL source et de la déclaration CC0 pour chaque ligne HPLT retenue.

## Limites et suite

Ce corpus élargit fortement le nombre de tokens disponibles, mais son ajout est principalement du texte web et ne constitue pas une validation scientifique du modèle. La sélection dépend d’un dérivé tiers de HPLT; les droits de chaque page source ne sont pas audités individuellement. Les contrôles automatiques de données personnelles et de contenu ne remplacent pas une revue humaine. Aucun contrôle de déduplication MinHash local n’a été appliqué. Le corpus reste `candidate_not_legal_clearance`; son utilisation dans EXP-0028 ne vaut pas autorisation de publier les poids.

La prochaine expérience devra comparer, avec architecture, tokenizer, budget d’optimisation, seed et benchmarks identiques, un modèle entraîné sur le mélange antérieur et un modèle entraîné sur ce mélange enrichi. Le rapport devra séparer la loss par langue et source, puis réserver les six benchmarks au test final.
