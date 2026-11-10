# EXP-0045 : pilote anglais sur SimpleStories

Statut : **entraîné localement pour étude exploratoire; aucun gain comparatif confirmé**.

Un pilote CPU a été mené sur le candidat A de 935 040 paramètres. Les poids n'ont pas été publiés. SimpleStories conserve le statut `candidate` dans le registre : sa carte déclare MIT, le papier attribue les textes à GPT-4o-mini, et la revue des termes de sortie et de la vie privée n'est pas terminée. Le run n'autorise donc pas à redistribuer un checkpoint.

## Corpus et reproductibilité

- Révision `SimpleStories/SimpleStories`: `e63b8adc3b1a1bdc7cac5b500d150b71346b0628`.
- Fichier source : premier shard train seulement, SHA-256 `ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4`.
- Échantillon déterministe, 60 000 histoires train et 3 108 dev. Le groupe `generation_id` reste affecté à une seule partition. Chaque ligne a un ID de génération, le modèle source, la ligne du shard et un hash du texte.
- Tokenizer byte-level BPE, 1 024 entrées, appris sur 30 000 documents du train seulement; SHA-256 `ab95761755cd2a0217a92a7b0c8123ddc81e16be0929c7613cbba0d46937e664`.
- La partition dev vient du même shard/source. Ce n'est pas une évaluation externe et elle ne mesure pas adéquatement la mémorisation.

Installation et préparation (PyArrow 26.0.1 utilisé ici) :

```bash
pip install -e .
python -m pip install pyarrow==26.0.1
hf download SimpleStories/SimpleStories --repo-type dataset \
  --revision e63b8adc3b1a1bdc7cac5b500d150b71346b0628 \
  data/train-00000-of-00007.parquet --local-dir data/raw/simple-stories-e63b8adc
PYTHONPATH=src python scripts/build_exp0045_simple_stories_pilot.py \
  --parquet data/raw/simple-stories-e63b8adc/data/train-00000-of-00007.parquet \
  --out data/processed/exp0045-simple-stories-pilot --max-train 60000
```

Les manifests archivés relient les shards tokenisés aux splits, le tokenizer, l'échantillon source et le code de préparation. L'usage exploratoire d'une source non approuvée est marqué dans le manifest. Cela ne modifie pas son statut dans le registre.

## Résultats d'entraînement

Les trois phases utilisent AdamW, séquences de 256 tokens, lots de quatre séquences, calcul float32 CPU et la même graine. Les phases 2 et 3 initialisent les poids du checkpoint précédent dans un nouveau run. L'optimiseur est réinitialisé à chaque phase; les journaux ne correspondent donc pas à une seule reprise avec état optimiseur continu.

| Phase | Étapes | Tokens présentés | Perte de validation initiale → finale | SHA-256 du checkpoint |
|---|---:|---:|---:|---|
| Initiale | 2 048 | 2 097 152 | 4,147 → 2,760 | `8e993d89e67acd22459e3492455fc5b57c16a350b8d66315948e3e5b9ca67ba5` |
| Continuation 1 | 8 192 | 8 388 608 | 2,709 → 2,391 | `4838ed0d933633d9a84263cf4d4b0aea66fce922fade295ed867b13dad5ba598` |
| Continuation 2 | 16 384 | 16 777 216 | 2,389 → 2,272 | `4c0ac79f1f43e1126ba406cede769526d63f45d68c4557976c04cfea926d06d7` |

Total : 27 262 976 présentations pour une seule graine. La baisse de perte indique que le réseau apprend à prédire cette partition; elle ne garantit pas une meilleure prose ni une bonne généralisation.

## Générations

Cinq préfixes identiques ont été donnés au checkpoint final et à `shibatch/tiny1m`, avec décodage glouton jusqu'à 120 tokens. Les sorties complètes et IDs sont dans `generation-review.jsonl`.

Le nouveau modèle produit parfois une petite suite d'événements, mais répète aussi des motifs comme « a small, shiny object » et des formules de trésor. `tiny1m` garde une structure de récit par endroits, avec des erreurs lexicales (« loyaar », « gare ») et des phrases incohérentes. Ces cinq exemples ont été inspectés directement, sans panel aveugle : **ils ne permettent pas de déclarer un vainqueur**.

## Artefacts

- `record.json` : budget, limites, hashes et trois phases.
- `corpus-manifest.json` et `tokenizer-manifest.json` : lignées des données.
- `phase*-run_manifest.json`, `phase*-metrics-*.jsonl` (en morceaux), `phase*-eval.jsonl`, `phase*-observability-summary.jsonl` : résumés par étape des statistiques par couche (moyenne et maximum; logs complets restent locaux).
- `generation-review.jsonl` : prompts, sorties et IDs de tokens des deux modèles.
- Les checkpoints de 11,3 Mo restent locaux et ne sont pas dans Git; leurs hashes sont archivés dans `record.json`.

Pas d'évaluation humaine aveugle, de deuxième graine, de mesure d'énergie CPU, de juge automatique, de distillation ni d'essai des candidats B/C. Ces expériences restent nécessaires pour déterminer si le modèle peut dépasser TinyStories sur des histoires inédites.
