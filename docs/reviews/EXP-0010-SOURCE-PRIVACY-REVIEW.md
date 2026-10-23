# EXP-0010 1M — dossier de revue des sources et de la confidentialité

**État au 2026-10-22 : paquet public sur Hugging Face; revues humaines des droits et de la confidentialité toujours non signées.**

Ce dossier rassemble les vérifications reproductibles disponibles pour le candidat v4 et sépare
leurs résultats des décisions qui relèvent du mainteneur. Il ne constitue ni un avis juridique, ni
une attestation de conformité, ni une autorisation de publier.

**Ces revues de droits et de confidentialité sont des contrôles de gouvernance propres au projet,
pas des signatures exigées par l’OSAID 1.0.** L’auto-évaluation OSAID, fondée sur le contenu et les
termes du paquet, est publiée séparément dans [OSAID-1.0-EXP0010-ASSESSMENT.md](OSAID-1.0-EXP0010-ASSESSMENT.md).

## Mise ? jour de l?audit HAL du 2026-10-22

Un addendum documente la v?rification actualis?e des champs de licence au niveau des fichiers :
[EXP-0010-HAL-FILE-LICENSE-ADDENDUM-2026-10-22.md](EXP-0010-HAL-FILE-LICENSE-ADDENDUM-2026-10-22.md).
L?audit pagin? a interrog? `licence_s` (fichier principal) et `fileLicenses_s` (fichiers documentaires)
pour toutes les versions HAL retourn?es. Il confirme les 3 450 identifiants d?j? accept?s et exclut
33 des 3 483 lignes d?clar?es CC0 par le dataset. Cela renforce la concordance des m?tadonn?es de
licence, sans v?rifier l?autorit? du d?posant ni remplacer la revue humaine des droits ou de la vie
priv?e. Le manifeste source public comprend d?sormais les titres, les auteurs d?clar?s, les dates et
les DOI disponibles. Les marqueurs de revue humaine restent inchang?s.

## Candidat couvert

| Élément | Valeur |
|---|---|
| Identifiant | `slm-tavenel-1m-exp0010-s421101` |
| Empreinte SHA-256 du checkpoint source de l’entraînement | `d86a5617dd08c7a43c9e6ac1a276f19455a82ecf2f49bdda55b2f5cbb1ddcfd8` |
| Empreinte SHA-256 des poids empaquetés `model.safetensors` | `b2ef1606aacc86808d575cc3ea85d93c689b37e9dfb933a6627ea7d8e6dbde6f` |
| Paquet local | `exports/slm-tavenel-1m-exp0010-release-candidate-v4/` |
| Source déclarée | `PleIAs/French-Science-Commons`, révision `bfa114c09d7c28dcd0a8a47fe445c64b4a3c6156`, sous-ensemble `french_open_science` |
| Corpus après sélection et filtrage | 1 229 documents : 748 anglais et 481 français |
| Empreinte du rapport HAL inclus | `e79fd757123cfdb2328b950bb23a4c698a8de18f676e7bd0d3349b79dc2d7abb` |
| Empreinte du rapport de filtrage de confidentialité inclus | `81304fa28f7836c2ec592284251bf863bb8a34cc0e59fbe7674be59b29cf4f25` |
| Empreinte du manifeste de provenance des 1 229 documents | `1b5facec126e3c9939ff501562ee022b55e15c13b8cb2aef3d38cb6dfddb1711` |
| Empreinte du rapport de parité d’inférence inclus | `29002711fbfce16e711b8ac2a3e561a5475f16a99493599d266f04521c3fa63d` |
| État du paquet dans `provenance.json` | `uploaded=true`; `source_rights_review_signed=false`; `manual_privacy_review_signed=false`; la page déclare explicitement les limites de droits et de confidentialité |
| Licence déclarée pour les poids, le code et les informations de données | MIT |

Les empreintes ci-dessus identifient le paquet local examiné. Toute modification des poids, de la
sélection de documents, des rapports ou des fichiers de provenance impose de refaire le lien entre
le dossier et le paquet.

## Ce qui a été vérifié automatiquement

### Provenance et métadonnées de licence

- L’audit paginé compare les 3 450 identifiants sélectionnés aux versions retournées par l’API HAL :
  1 767 notices anglaises et 1 683 françaises. Le rapport donne 3 450 correspondances CC0, aucune
  notice absente et aucun conflit de licence dans l’instantané audité.
- La sélection écarte les notices sans correspondance ou avec une licence différente ou contradictoire.
- Le paquet contient désormais `training_record/source-record-manifest.jsonl`, avec une ligne par
  document retenu, ainsi que `training_record/hal-license-audit.json`,
  `training_record/source-registry.yaml`, `training_record/data-decisions.jsonl` et les manifestes
  de pipeline, de corpus et de s?paration. Le manifeste donne les identifiants/URI, le titre, les
  auteurs d?clar?s, la date, le DOI lorsqu?il est fourni, la licence d?clar?e, la langue, le split
  et les empreintes avant/apr?s traitement; il ne contient pas le texte des articles.
- Le manifeste contient 1 229 entrées. Son SHA-256 est inscrit dans l’inventaire `provenance.json`.
- Les empreintes des rapports HAL et confidentialité du paquet correspondent à celles inscrites
  dans `provenance.json`. L’empreinte des poids empaquetés correspond à l’entrée
  `files_sha256["model.safetensors"]`. L’empreinte du checkpoint source est une valeur distincte :
  la conversion en `safetensors` change le fichier, et les deux valeurs sont conservées.
- Le manifeste mis ? jour reprend les m?tadonn?es fournies par la source : 1 229 titres, 1 229
  champs auteur non vides, 1 229 dates de publication et 380 DOI. Les 849 DOI absents restent
  absents, sans inf?rence. C?est une v?rification des champs fournis par la source, pas une
  validation ind?pendante de l?identit? des auteurs ni de leur qualit? de titulaires de droits.

**Limite décisive :** l’audit établit ce que les métadonnées du dépôt et de HAL déclarent, pas qui a
appliqué CC0 ni si cette personne détenait les droits nécessaires. CC0 précise lui-même qu’il est
appliqué par l’affirmant dans la mesure où celui-ci détient les droits concernés. Les conditions
CC0 ne peuvent donc pas transformer une métadonnée en preuve d’autorité du déposant.

La page publique du corpus présente aussi des enregistrements sous des licences autres que CC0.
Cela confirme que la licence globale du jeu n’est pas une preuve suffisante pour chaque article et
que la sélection par notice doit rester documentée. Le rapport actuel fournit cette sélection et
la comparaison HAL; la vérification de l’autorité des déposants reste à décider.

### Filtrage automatisé et examen d’échantillons

Sur les 1 229 documents retenus, le rapport de confidentialité indique zéro caractère de
remplacement, zéro chaîne de courriel détectée, zéro document correspondant aux termes de risque
configurés et zéro résultat du détecteur résiduel configuré. Les règles ont auparavant supprimé
des documents signalés par des filtres d’intégrité, de contenu sensible et de qualité; elles ont
aussi masqué les occurrences reconnues de courriels, téléphones et motifs IP.

Un échantillon déterministe de 10 courts extraits par langue (20 au total) a été examiné. Le rapport
signale des signatures publiques attendues et ne relève, dans ces extraits, ni récit de cas centré
sur un patient ni contact non public. Les identifiants et empreintes de l’échantillon sont conservés
dans le rapport embarqué; les extraits et le texte des articles ne sont pas reproduits ici.

Ces résultats constituent un filtrage technique et un examen de petit échantillon. Ils ne couvrent
pas toutes les formes de données personnelles, les noms ou faits rares, les informations
identifiantes dans les tableaux et références, ni les informations que les poids pourraient
mémoriser. La CNIL distingue l’anonymisation de la pseudonymisation et décrit l’anonymisation comme
une analyse du risque de ré-identification, pas comme la simple suppression de motifs directs.
Le projet ne qualifie donc pas ce corpus d’anonyme.

## Évaluation actuelle des portes de publication

| Porte | État vérifiable | Ce qui manque pour la décision |
|---|---|---|
| Corpus exact et traçabilité technique | Documentés par révision, sélections, règles, rapports et empreintes | Revalider ces empreintes si le paquet change |
| Déclaration CC0 par notice | 3 450/3 450 déclarations concordantes avec les versions retournées par HAL dans l’instantané | Revue humaine d’un échantillon de notices et des dépôts d’origine; décider du niveau d’incertitude acceptable sur l’autorité des déposants |
| Confidentialité automatisée | Contrôles configurés sans résultat résiduel sur le corpus retenu | Le résultat n’équivaut pas à une revue humaine exhaustive |
| Confidentialité humaine | Échantillon de 20 extraits consigné | Une personne responsable doit consigner le périmètre examiné, les exclusions, le risque résiduel et sa décision sur le corpus exact |
| Licence MIT des poids et du code | Fichier MIT présent dans le paquet | Cette licence décrit les droits accordés sur le paquet du modèle; elle ne valide pas les droits sur les sources |
| Publication Hub | Paquet public à `https://huggingface.co/athenista/slm-tavenel-1m-exp0010`; parité d’inférence marquée `passed`; provenance du paquet: `uploaded=true` | Les revues humaines des droits et de la confidentialité restent non signées. La disponibilité publique est un fait distinct de ces revues; elle ne vaut ni clearance juridique ni validation exhaustive de confidentialité |
| Auto-évaluation OSAID 1.0 | Le paquet préparé satisfait, selon le projet, les exigences centrales sur les libertés, les informations de données, le code et les paramètres; MIT est déclaré | Le paquet doit être publiquement disponible avant d’être une offre Open Source AI effective. L’OSI ne certifie pas les modèles individuels; voir l’auto-évaluation liée ci-dessus |

Le paquet est effectivement public. Selon l’auto-évaluation du projet, il satisfait les exigences
centrales OSAID 1.0 sur les éléments et les termes fournis; ce n’est pas une certification de l’OSI.
L’incertitude sur l’autorité
derrière une déclaration CC0 reste une limite de provenance et de risque juridique à divulguer;
elle ne constitue pas une exigence de signature OSAID. L’OSI indique que sa définition ne prescrit
pas un instrument légal précis pour les paramètres et que l’OSI ne certifie pas les modèles
individuels.

## Décision du mainteneur à consigner

Compléter ce tableau après examen du candidat identifié ci-dessus. Une décision « conditionnelle »
doit nommer les conditions et préciser si la publication est autorisée avant leur réalisation.

| Champ | À compléter par le mainteneur |
|---|---|
| Nom et rôle du réviseur | En attente |
| Date de revue | En attente |
| Empreintes du checkpoint source, des poids empaquetés, du manifeste et des rapports examinés | En attente de confirmation |
| Source : portée de l’examen (notices, dépôts, licences déclarées) | En attente |
| Source : décision et incertitudes acceptées | En attente |
| Confidentialité : portée de l’examen humain | En attente |
| Confidentialité : exclusions supplémentaires et risques résiduels | En attente |
| Décision finale — refusée, conditionnelle ou approuvée | En attente |
| Autorisation de publication publique | Contrôle de gouvernance du projet, distinct de la conformité OSAID |
| Signature ou lien vers la décision signée | En attente |

Ne pas modifier les marqueurs `*_signed` ou `public_release_approved` du candidat avant d’avoir
consigné une véritable décision du mainteneur portant sur ces empreintes. Si le corpus change,
reconstruire le tokenizer, les séparations et le modèle, puis refaire les contrôles liés aux
empreintes.

## Références primaires

- [Fiche du jeu PleIAs/French-Science-Commons](https://huggingface.co/datasets/PleIAs/French-Science-Commons) — présente le corpus agrégé et ses champs de licence par enregistrement.
- [Documentation officielle de l’API HAL](https://api.archives-ouvertes.fr/docs/search) — décrit l’API de recherche utilisée pour la comparaison des métadonnées.
- [Code juridique CC0 1.0 en français](https://creativecommons.org/publicdomain/zero/1.0/legalcode.fr) — précise la portée de CC0 et la responsabilité de l’affirmant quant aux droits détenus.
- [Définition OSAID 1.0](https://opensource.org/ai/open-source-ai-definition) — spécifie les libertés, les paramètres, le code et les informations sur les données.
- [CNIL : l’anonymisation des données personnelles](https://www.cnil.fr/fr/technologies/lanonymisation-de-donnees-personnelles) — distingue anonymisation et pseudonymisation et décrit l’évaluation du risque de ré-identification.
