# Ramenez la coupe à la maison
## Download dataset
+ https://www.kaggle.com/c/12500/download-all
+ Mettre toutes les données dans \jigsaw-unintended-bias-in-toxicity-classification
+ Ne mettre que des chemins relatifs dans les scripts
___
## Listes d'idées à traiter :
- [x] Commencer un modèle de Benchmark de base
- [ ] Modifier Vectorize pour inclure le dependency tree
- [ ] Ajouter les données test dans la vectorisation
- [ ] Attention à la forme de la prédiction finale (catégories vs. float) comme la métrique finale utilise des ROC
  - [ ] Tester avec des valeurs discrètes en sortie
  - [ ] Tester avec des valeurs continues en sortie
- [ ] Les minorités ne sont pas exclusives (e.g. black + homosexuel + mental illness ) 
> "*Je suis un '__orientation sexuelle__' '__handicap mental__' et je déteste les '__religion__' (et les) '__race__'.*"
- [ ] Emojis
  - [ ] Vérifier si il y a des emojis. 
  - [ ] Les encoder si besoin est. 
 
___

+ https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec


