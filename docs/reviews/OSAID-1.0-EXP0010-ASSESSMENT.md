# SLM-Tavenel EXP-0010 — auto-évaluation OSAID 1.0

**Conclusion du projet : le paquet publié satisfait, à notre connaissance, les éléments centraux
de la définition OSAID 1.0 pour un modèle et ses poids.** Ce constat est une auto-évaluation du
projet, pas une validation ni une certification de l’OSI. Le paquet est public sur
[Hugging Face](https://huggingface.co/athenista/slm-tavenel-1m-exp0010), révision
`3bb07398ec8105d9625cf8e346425b9987c65c48`.

## Exigences centrales de la définition

| Exigence | État | Preuve dans le paquet EXP-0010 seed 421101 |
|---|---|---|
| Liberté d’utiliser, étudier, modifier et partager | Satisfaite par les termes proposés pour le paquet | `LICENSE` MIT couvre les poids, le code et les informations de données; aucune condition d’usage additionnelle propre au modèle n’est ajoutée. MIT est une licence approuvée par l’OSI. |
| Informations détaillées sur toutes les données d’entraînement, sous termes approuvés | Satisfaite | `training_record/data-card.md`, `data-pipeline.yaml`, `data-decisions.jsonl`, `data-manifest.json`, `source-record-manifest.jsonl`, `hal-license-audit.json`, `split-manifest.json`, `tokenizer-manifest.json` et `token-manifest.json` décrivent les sources, la sélection, les filtres, le traitement, les langues, les séparations et les empreintes. La licence déclarée pour ces informations est MIT. |
| Origine et emplacement des données publiques ou tierces | Satisfaite par divulgation avec liens | Les informations nomment la révision exacte de `PleIAs/French-Science-Commons`, le sous-ensemble, les identifiants des 1 229 documents retenus et leurs liens HAL. Le texte source reste chez les dépôts publics et n’est pas recopié dans le paquet. Aucun jeu de données payant ou non public n’est déclaré comme source d’entraînement. |
| Code complet pour préparer, entraîner, évaluer et exécuter le modèle | Satisfaite | Le paquet comprend l’instantané du code d’entraînement, les scripts de sélection/filtrage, de séparation, de tokenisation, les configurations, l’évaluation, le code d’inférence Transformers et les rapports de réplication. Les fichiers sont sous MIT. |
| Architecture et paramètres du modèle | Satisfaite | `hf_model.py`, `hf_config.py`, `config.json` et `model.safetensors`; poids sous MIT. Le manifeste conserve le SHA-256 des 228 fichiers inventoriés, hors `provenance.json`. |

Les sources disponibles publiquement sont traitées dans les informations de données comme des
données tierces consultables en ligne. Les enregistrements retenus portent des déclarations CC0
dans le corpus et dans les métadonnées HAL; l’audit compare 3 450 enregistrements à toutes les
versions retournées par l’API HAL. Cette déclaration de provenance est reproduite fidèlement,
sans prétendre avoir vérifié l’identité ni l’autorité de chaque déposant. Cette incertitude est
rendue visible aux utilisateurs; elle n’est pas remplacée par une fausse attestation.

## Ce qui n’est pas une condition OSAID

Les signatures internes de revue juridique, la certification de confidentialité et une autorisation
de publication signée ne sont pas des cases exigées par la définition OSAID. Elles restent utiles
pour documenter la gouvernance et les risques du projet, mais elles ne doivent pas être présentées
comme des critères de conformité OSAID. La FAQ de l’OSI sépare explicitement les exigences de
sécurité/responsabilité de la définition et permet d’utiliser des données publiques ou non
partageables si leur description répond aux exigences de transparence.

De même, l’OSI ne certifie pas les modèles individuels. Il est donc correct de déclarer une
**auto-évaluation de conformité OSAID 1.0** avec ses limites; il serait incorrect de dire que
l’OSI a certifié ou approuvé SLM-Tavenel.

## État de publication

Le paquet, sa carte, `LICENSE`, `NOTICE.md`, sa provenance et le présent auto-audit sont publics
sous MIT à la révision Hub indiquée ci-dessus. Le dépôt marque les limites de provenance et de
confidentialité; la publication ne les résout pas et ne vaut pas certification OSI.

## Références

- [Définition OSAID 1.0](https://opensource.org/ai/open-source-ai-definition)
- [FAQ OSAID — catégories de données, responsabilités et certification](https://opensource.org/ai/faq)
- [Checklist OSI pour les systèmes d’apprentissage automatique](https://opensource.org/ai/checklist) — document d’aide à l’interprétation; l’OSI précise qu’il ne s’agit pas d’un manuel d’évaluation opérationnel.
- [Licence MIT sur le site de l’OSI](https://opensource.org/license/MIT)
- [Dossier des sources et de la confidentialité EXP-0010](EXP-0010-SOURCE-PRIVACY-REVIEW.md)
