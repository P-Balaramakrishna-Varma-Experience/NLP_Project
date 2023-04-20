# NLP_Project
dict_keys(['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc'])
- enhanced depenecnies?
- subtype dependencies?
- Deal with padding? [nothing to be done!!! just keep in mind]

- droput
- initiaziion paramters

## optimization
- multigpu
- pytorch 2.0

## loss (root modeling and padding)
have visulaiexe spit into two parts
    - do not include loss for padded words
    - do not include loss for root word
- root word progression throught the system  (verify) -- [two option biltml layer expriment and decide]

- test accuryise properly