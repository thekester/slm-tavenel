# EXP-0045 : histoires anglaises sous un plafond strict de paramètres

Statut : **pilote CPU exploratoire terminé; aucune victoire sur TinyStories établie**.

Cette expérience reprend le diagnostic EXP-0044 et remplace sa cible prospective de 1,18 M et son mélange bilingue. L'objectif utilisateur est désormais la meilleure génération narrative possible avec au plus 1 M de paramètres au total. Pour égaler précisément la référence utilisée dans les benchmarks, les candidats sont plafonnés à **935 040 paramètres uniques**. Le calcul et les données d'entraînement ne sont pas des budgets imposés, mais restent mesurés.

La [revue détaillée](../docs/research/stories-1m-2026-11-10.md) rassemble corpus, révisions, sources scientifiques, limites de transfert et exigences de traçabilité. Le [dossier du pilote](../docs/reviews/exp0045-story-pilot-2026-11-10/README.md) conserve les manifests, hashes, journaux d'entraînement et sorties comparatives.

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

Le décompte exact des trois configurations est vérifié par les contrôles antérieurs. Le candidat A a été entraîné en trois phases CPU et une graine, pour **27 262 976 tokens présentés**. Sa perte de validation intra-corpus est passée de 4,147 au premier contrôle à 2,272 à l'évaluation finale. Cela confirme l'apprentissage de cette partition, pas une amélioration sur des histoires inédites.

L'inspection descriptive de cinq préfixes comparés à `tiny1m` montre que l'élève produit quelques séquences narratives, mais répète encore des formules et dérive du préfixe. Ces exemples ne constituent pas un test aveugle ou une preuve statistique. Le modèle n'a pas démontré qu'il bat TinyStories.

Le tokenizer, le corpus filtré, les hashes, les trois manifests de run et les métriques d'observabilité sont archivés. Le corpus est un échantillon d'un seul shard : 60 000 histoires train et 3 108 dev. Il reste candidat, sous revue, et n'est pas autorisé pour publier des poids. Le checkpoint reste local; son SHA-256 est enregistré dans le record.

Pour vérifier les comptes localement :

```bash
pytest -q tests/test_fluency.py
```

Les anciennes configurations à environ 1,18 M restent des propositions historiques hors de ce protocole. Aucun résultat de génération ne leur est attribué.

## Correctifs après revue de la PR (10 novembre 2026)

**Séparation train/dev.** Les `generation_id` de SimpleStories ont la forme `<hash de complétion>-<numéro>`, et une complétion contient plusieurs histoires écrites depuis le même prompt. Le pilote sépare sur l'identifiant complet : **516 histoires dev sur 3 108 (16,6 %) ont une histoire de la même complétion dans le train**, mesuré sur le shard épinglé. Aucune histoire dev n'a un texte normalisé identique dans le train. Les pertes de validation du pilote restent des mesures intra-corpus, sans valeur de généralisation. `scripts/build_exp0045_simple_stories_v2.py` sépare sur le hash de complétion (60 000 train, 3 185 dev, aucune complétion partagée). Le script d'origine et le dossier du pilote restent inchangés, car leurs hashes sont enregistrés.

**Comparaison des générations.** Les cinq continuations du pilote sont gloutonnes et limitées à 120 tokens, contrairement à la règle d'EXP-0044 (320 octets, échantillonnage). Elles font 351 à 392 octets pour le candidat et 150 à 334 pour `tiny1m`. Elles ne doivent pas servir à classer les modèles. La comparaison est à refaire avec `scripts/compare_fluency.py`, sur un candidat exporté par `slm.export` et avec `configs/eval/fluency-stories-en-v1.jsonl`. Ces cinq préfixes sont publics : ils servent au développement, pas de test final.


## Report effectif des correctifs sur cette branche

Les fichiers du correctif local sont maintenant int?gr?s sur la branche de la PR : split par famille de compl?tion, cinq pr?fixes publics de d?veloppement et configuration du run continu de 100 M de tokens. Les archives du premier pilote CPU ne sont pas modifi?es. Un test de r?gression v?rifie que deux histoires d'une m?me compl?tion restent dans la m?me partition.

Un run distinct depuis z?ro, seed 421201, a termin? 6 104 ?tapes sur le split corrig? (60 000 train, 3 185 dev, z?ro groupe de compl?tion partag?). Son export standard Llama de 935 040 param?tres est public ? https://huggingface.co/athenista/slm-tavenel-story-935k, par autorisation explicite du propri?taire avec les r?serves amont document?es. Le statut de la source reste candidat; les anciennes d?clarations concernant les poids du pilote CPU restent des ?l?ments historiques de ce pilote.

Sur les m?mes textes SimpleStories dev, cet export mesure 0,937012 BPB contre 1,670289 pour le snapshot tiny1m. Cette mesure favorise son domaine d'entra?nement et ne d?montre aucune sup?riorit? narrative. L'?valuation TinyStories et la revue humaine sont suivies s?par?ment dans le d?p?t principal. Le code et les donn?es de ce premier pilote n'ont pas ?t? remplac?s par ceux du nouveau run.
