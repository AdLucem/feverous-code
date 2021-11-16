# Section-Level Classifier

To classify sections, for a given claim, into relevant and non-relevant.

## Representing a Section

- Using TF-IDF search and entity matching, top `N` sentences from a given section are chosen.
- The lines immediately before and after each chosen sentence are also chosen. 

So for eg, for a given section:

Relevant lines: `sentence_1, sentence_5, sentence_6, sentence_11`
Representation to feed into RoBERTa:

```
sentence_1, sentence_2, claim | label
sentence_4, sentence_5, sentence_6, sentence_7, claim | label
sentence_10, sentence_11, sentence_12, claim | label
```

With the label being 1 if 'relevant' (in the gold evidence set) or 0 if 'non-relevant' (not in the gold evidence set)

So for each claim, the datapoint will be a JSON in format:

- claim_id
- claim
- sections
    - [(section_id, label)]
- sentences
    - [{"section": section_id, 
        "sequence":[sentence_sequence],
        "label": label} for each section, sentences pair]
