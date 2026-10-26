# Corpus bilingue filtré, candidat v1

**Date de construction :** 2026-10-24
**État :** corpus local reproductible; les étiquettes de sources et licences restent celles fournies par les dépôts. Ce rapport distingue les vérifications techniques de l'examen juridique.
**Aucun nouvel entraînement n'a été lancé.**
## Jeux Hugging Face utilisés

Le corpus réunit trois sources avec des niveaux de provenance différents. Les identifiants, révisions, textes sources, labels, transformations et empreintes sont conservés dans les fichiers de données et les registres d'attribution locaux.

1. **[PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons/tree/bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156)**, sous-ensemble CC0 déjà filtré par le pipeline FSC v4 du projet. Il apporte l'anglais et le français. L'audit HAL et le filtre technique préexistants sont conservés. La revue humaine des droits et de la confidentialité reste non signée.
2. **[OpenBallast/ballast-t2](https://huggingface.co/datasets/OpenBallast/ballast-t2/tree/e53b6dd0c2fb6a16909e30f6fdb2c5d787878cdf)**, uniquement sa partition anglaise CC BY. Pour chacun des 61 livres retenus, le constructeur vérifie le fichier `LICENSE` au commit complet référencé dans les métadonnées. Il exclut la partition NC-SA. L'attribution par livre, le dépôt source, le commit et l'empreinte du fichier `LICENSE` sont conservés.
3. **[PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus/tree/307910e4c5d040d6f318e6edf2a2b97849155771)**, un seul fichier du dataset : `common_corpus_1/subset_100_1.parquet`, SHA-256 `4ee719a130b7f86978b08c20cc9f490309e2bae2a5c0df4f04fd427365530f07`. Ce fichier contient 69 907 enregistrements dans cette révision. Le constructeur ne prétend pas avoir traité Common Corpus au complet.

Pour Common Corpus, la règle retient des documents écrits en anglais ou en français dans cinq couples collection/étiquette upstream :

| Collection | Étiquette conservée telle quelle | Documents retenus |
|---|---|---:|
| OpenAlex | `CC-By` | 96 |
| Eurlex | `CC-By` | 97 |
| French-Science-Pile | `Various open science` | 203 |
| govinfo | `Public Domain` | 120 |
| dotgov | `Public Domain` | 157 |
| **Total Common Corpus** |  | **673** |

Les étiquettes Common Corpus sont des métadonnées de provenance, pas une confirmation indépendante des droits : `CC-By` ne précise pas la version, `Various open science` ne désigne pas une licence unique et `Public Domain` reste l'affirmation du jeu. Chaque ligne retenue garde la valeur exacte, la collection, l'identifiant source, le titre, le créateur lorsqu'il est fourni, la date et la langue. Les autres collections et étiquettes du shard sont exclues par la règle de sélection; elles ne sont pas une raison pour écarter l'ensemble de Common Corpus.

Pour OpenBallast, la carte décrit les dépôts OpenStax épinglés, l'exclusion des titres relicenciés NC-SA et l'attribution par livre. La page générale actuelle d'OpenStax indique CC BY-NC-SA, tandis que les commits historiques retenus par OpenBallast déclarent CC BY 4.0. Le constructeur vérifie les fichiers `LICENSE` à ces commits historiques et conserve leur SHA-256. Cela fournit une trace vérifiable, sans constituer une confirmation directe du titulaire pour chaque élément tiers éventuellement inclus. Voir aussi la [politique de licence OpenStax](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks).

## Filtres et séparation

- vérification des SHA-256 des fichiers sources, des six benchmarks et du tokenizer;
- OpenBallast : licence exacte `CC-BY-4.0`, source, commit et URL concordants, puis vérification des 61 fichiers `LICENSE` officiels;
- Common Corpus : allowlist explicite de collection, étiquette de licence upstream, langue EN/FR et modalité `Written`; l'étiquette originale est conservée;
- exclusion des textes trop courts, très longs, bruités, avec caractères de remplacement ou faible proportion de texte alphanumérique;
- retrait des enregistrements contenant les motifs configurés d'emails, téléphones, IP contextualisées et numéros sensibles contextualisés; pour Common Corpus, retrait supplémentaire de termes de contenu humain/sensible;
- déduplication exacte entre toutes les sources;
- exclusion d'un document si une séquence normalisée exacte de 13 tokens apparaît dans une question ou un choix des six benchmarks locaux épinglés;
- splits OpenBallast groupés par commit de dépôt, FSC v4 conserve ses splits par document, Common Corpus est réparti de manière déterministe par collection et identifiant.

Le contrôle a trouvé **zéro chevauchement exact de 13 tokens** avec les benchmarks. Cela ne prouve pas l'absence de toute contamination. Sur le shard Common Corpus, 67 549 enregistrements n'ont pas correspondu à l'allowlist, 902 ont été retirés sur motifs directs de coordonnées, 192 sur signaux de contenu sensible et 591 sur les autres filtres. Chaque décision est journalisée sans copier le texte dans le journal.

## Résultat construit

Les tokens sont comptés avec le tokenizer SLM-Tavenel v4, SHA-256 `93c7b9364de2c42315571c851353862e63830bfbeb220dc14dde80e6ca82cfcb`.

| Split | Documents | OpenBallast | FSC v4 | Common Corpus | Tokens | EN tokens | FR tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 61 435 | 59 860 | 1 041 | 534 | 41 274 981 | 34 713 376 | 6 561 605 |
| Développement | 8 718 | 8 580 | 72 | 66 | 4 208 719 | 3 816 997 | 391 722 |
| Test | 7 461 | 7 272 | 116 | 73 | 5 255 076 | 4 349 348 | 905 728 |
| **Total** | **77 614** | **75 712** | **1 229** | **673** | **50 738 776** | **42 879 721** | **7 859 055** |

Le train contient **15,9 % de tokens français**; la recette reste donc majoritairement anglaise. Par source, les tokens du train sont 18,25 M OpenBallast, 15,98 M FSC et 7,05 M Common Corpus. Avant entraînement, fixer le mélange voulu et rapporter les résultats séparément par langue et source.

Les fichiers compressés, le manifeste complet, le journal des décisions et les deux registres d'attribution sont dans `data/processed/slm-tavenel-reliable-corpus-candidate-v1/` (répertoire local ignoré par Git). Le registre Common Corpus conserve aussi les labels génériques au lieu de leur attribuer une version de licence supposée.

| Artefact | SHA-256 |
|---|---|
| train | `a862ac1311ee65b34b53bdf8bd9b6901a7c2cf2bca6d28ce64ba0f8e57084e82` |
| dev | `727a7c40dc3a76ad6abf75bc4f8ea5c5852559fd34655023ded2b25d84ca48a1` |
| test | `9982c6482b3022337628011e60954acca785ac001f81e48774f9479fa0a83796` |
| décisions | `830188f05eca0288a335b57c8dadc0ec281b6dd95e56ac42013fcc2fdf13198d` |
| attribution OpenBallast | `4beca3d0e04cfa0ee72ea7d9fca7e07bfcde9a9b046c75bd8cfb7fb01398c3a0` |
| attribution Common Corpus | `069018038748a752bc887a957d8a86c24fc1bc1628cde93cd1930943ac2acb84` |

## Portée et limites

Ce corpus est techniquement filtré et traçable, mais pas déclaré juridiquement validé. Les revues humaines des droits et de la confidentialité FSC restent non signées. Les licences déclarées par Common Corpus sont préservées comme telles, y compris leurs libellés génériques; la vérification technique n'en fait pas une preuve indépendante des droits. Les filtres automatisés ne constituent ni une anonymisation ni une revue exhaustive de confidentialité.

OpenBallast est exclusivement du texte pédagogique anglais; FSC apporte le français scientifique; Common Corpus apporte une petite sélection scientifique, gouvernementale et législative. Ce n'est pas un corpus généraliste équilibré. Le corpus n'a pas été converti en shards d'entraînement ni utilisé pour un entraînement. Le nombre de tokens dans le manifeste sert au comptage; il ne modifie pas les historiques ou résultats des modèles existants.

## Reproduction

Installer l'extra de curation (`pip install -e ".[data-curation]"`, qui épingle `pyarrow==19.0.1`), puis lancer :

```powershell
python scripts/build_reliable_corpus_candidate.py
```

Le script télécharge les fichiers épinglés s'ils sont absents, vérifie les empreintes et reconstruit les splits, décisions, registres d'attribution et manifeste. Pour réutiliser les fichiers de licence OpenStax déjà archivés :

```powershell
python scripts/build_reliable_corpus_candidate.py --skip-license-fetch
```
