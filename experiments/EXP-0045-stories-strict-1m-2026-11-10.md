# EXP-0045 : histoires anglaises sous un plafond strict de paramètres

Statut : **revue scientifique et configurations préparées, aucun entraînement effectué**.

Cette expérience reprend le diagnostic EXP-0044 et remplace sa cible prospective de 1,18 M et son mélange bilingue. L'objectif utilisateur est désormais la meilleure génération narrative possible avec au plus 1 M de paramètres au total. Pour égaler précisément la référence utilisée dans les benchmarks, les candidats sont plafonnés à **935 040 paramètres uniques**. Le calcul et les données d'entraînement ne sont pas des budgets imposés, mais restent mesurés.

La [revue détaillée](../docs/research/stories-1m-2026-11-10.md) rassemble corpus, révisions, sources scientifiques, limites de transfert et exigences de traçabilité.

## Candidats prêts à instancier

| ID | Configuration | Paramètres | Statut |
|---|---|---:|---|
| A | `configs/model/slm-story-935k.yaml` | 935 040 | Dense, premier candidat |
| B | `configs/model/slm-story-913k-deep.yaml` | 912 672 | Dense, plus profond et étroit |
| C | `configs/model/slm-story-921k-v2048.yaml` | 920 960 | Dense, vocabulaire plus grand |

Ces comparaisons portent sur des recettes architecturales complètes. Les variations de portes, normes QK, profondeur et FFN interdisent d'attribuer un résultat à un seul facteur sans contrôles supplémentaires. Chaque candidat conserve l'observation native, demande un nouveau tokenizer et un entraînement depuis zéro. Les fichiers de poids publiés ne sont pas compatibles par simple modification de YAML.

## Séquence d'expériences

1. Vérifier les sources et construire splits, groupes de doublons et tokenizers train-only. Ne pas changer les statuts du registre tant que les éléments nécessaires ne sont pas documentés.
2. Candidat A : SimpleStories anglais seul, TinyStories GPT-4 clean seul, mélange 70/30 par octets. Ce sont des propositions de recettes, pas des corpus déjà construits ou approuvés.
3. Paliers indicatifs 100 M, 300 M puis 1 Md de tokens présentés, avec validation narrative et suivi de l'exposition textuelle. Adapter la suite aux courbes; aucun nombre d'époques optimal n'est supposé.
4. Sur la meilleure recette, comparer A/B/C, puis AdamW/Muon, puis absence/présence d'un professeur narratif de 10 à 30 M. Valider l'identité complète du tokenizer avant toute KL.
5. Répéter le résultat retenu sur trois graines : 421201, 421202, 421203. Réserver récurrence, prédiction multiple et unlikelihood aux expériences suivantes, avec ablations isolées.

Référence principale figée : `shibatch/tiny1m@fdacad8ff9dc27bd72c77d3684075a494be1ba11`. Références secondaires : Tavenel publié et TinyStories original, avec leurs vrais totaux de paramètres.

## Critère d'adoption proposé

- 500 préfixes nouveaux hors entraînement et hors sélection d'hyperparamètres, stratifiés par difficulté narrative.
- Préférence humaine à l'aveugle, égalités valant 0,5, ordre A/B randomisé. Une paire par préfixe pour l'analyse principale, sélectionnée selon un plan équilibré entre graines. Les sorties supplémentaires sont conservées.
- Cible à préenregistrer : au moins 60 % de préférences et borne basse de l'intervalle à 95 % supérieure à 50 %, sans régression majeure de cohérence ou de répétition. C'est une cible, pas une mesure obtenue.
- Rapporter séparément grammaire, causalité, personnages, contraintes, diversité, mémorisation, bits/octet et troncatures. Les QCM généraux restent secondaires.
- Préfixes textuels identiques; limites de contexte natives visibles; analyse complémentaire d'histoires courtes accessibles à toutes les références. Pas de sélection des meilleurs échantillons ni de réécriture par un grand modèle.

## Validation disponible et travail restant

Le décompte exact des trois configurations est vérifié par `tests/test_fluency.py`. Les 54 tests ciblés passent; les trois architectures produisent des logits finis lors d'un passage avant sur entrée synthétique. Les sources, leur statut, leurs révisions et leur provenance sont documentées dans la revue; les nouveaux corpus et le test final restent à construire. Aucun pilote, professeur, mécanisme de récurrence ou nouveau poids n'est livré comme entraîné.

Pour vérifier les comptes localement :

```bash
pytest -q tests/test_fluency.py
```

Les anciennes configurations à environ 1,18 M restent des propositions historiques hors de ce protocole. Aucun résultat de génération ne leur est attribué.
