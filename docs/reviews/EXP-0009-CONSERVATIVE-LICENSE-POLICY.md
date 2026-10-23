# Conservative source-license handling for EXP-0009

**Policy selected:** when source records carry conflicting, missing, or version-ambiguous license evidence, use the most restrictive known evidence for screening. If there is no reliable permission for the exact source file, treat the item as not eligible for a new release-oriented training corpus until verified.

The attached `training_record/conservative-source-exclusions.jsonl` applies that policy to all 155 source records still unresolved in HAL as of 23 October 2026. It preserves per-record source ID, dataset declaration, current HAL outcome, DOI, and Crossref URLs where found. The 15 nonmatching Crossref declarations are separately discussed in the deep triage report.

This is an **exclusion policy for corpus construction**, not a license applied to the 25M weights. It does not remove those records from historical training or alter the model's learned parameters. The EXP-0009 checkpoint was already trained on these records (and on 31,262 additional records whose metadata labels match HAL but whose depositor authority and historical file versions were not independently verified). Therefore the list does not decontaminate or clear the existing 25M checkpoint.

CC BY-NC-ND 4.0 is the most restrictive explicit CC license observed among the queried declarations: it prohibits commercial use and sharing adaptations. CC BY-SA also has a share-alike condition. Those terms cannot be imposed as an overall model license while still claiming the OSAID freedoms to use for any purpose, modify, and share. Applying the more restrictive evidence record-by-record means excluding uncertain material from a future training set, not relabeling the aggregate checkpoint. See the [CC license comparison](https://creativecommons.org/share-your-work/cclicenses/) and [OSAID 1.0](https://opensource.org/ai/open-source-ai-definition).

## Status

The 155 unresolved records are excluded from any future release-eligible retraining corpus by this policy. The remaining 31,262 HAL metadata matches are not automatically cleared: exact historical file identity and depositor/rightsholder authority remain unverified. A fully conservative new corpus must include only independently verified records or must preserve the unresolved rights limitation explicitly. No rights review or publication approval is signed by this policy.
