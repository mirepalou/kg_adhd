# kg_adhd

## 1. Gathering data
The data is gathered using the BioKnowledge Reviewer: https://github.com/NuriaQueralt/bioknowledge-reviewer.

- The jupyter notebook with the specific functions used can be found on the ``code`` folder under the name ``bkr_adhd.ipynb``. 
- The only modification of the BioKnowledge Reviewer was on file ``monarch.py``, which can also be found in the ``code`` folder.

The output of this step are the nodes and edges for both diseases' knowledge graphs. The files can be found in the folder ``data``:
- HD: ``graph_nodes_v2024-05-27.csv`` (nodes) and ``edges_hd.zip`` (edges)
- AD: ``alz_graph_nodes_v2024-05-27.csv`` (nodes) and ``edges_ad.zip`` (edges)

## 2. Building the Knowledge Graphs
- neo4j latest version 5.16 
- through linux
- necessary plugins (apoc)
- commands:
  
``LOAD CSV WITH HEADERS FROM 'file:///graph_nodes_v2024-02-07.csv' AS row CREATE (:Node { id: row.id, semantic_groups: row.semantic_groups, preflabel: row.preflabel, synonyms: row.synonyms, name: row.name, description: row.description }); ``

``LOAD CSV WITH HEADERS FROM 'file:///edges_copy.csv' AS row MATCH (subject:Node {id: (row.subject_id)}), (object:Node {id: (row.object_id)}) WITH row, subject, object WHERE NOT row.property_label IS NULL AND trim(row.property_label) <> '' CALL apoc.create.relationship(subject, row.property_label, {}, object) YIELD rel SET rel += { reference_uri: row.reference_uri, reference_supporting_text: row.reference_supporting_text, reference_date: row.reference_date, property_description: row.property_description, property_uri: row.property_uri }; ``

## 3. Exploring the Knowledge graphs
The graphs are explored using Neo4j and python tools. For the latter we provide the jupyter notebook with the methods used in the ``code`` folder under the name ``exploration.ipynb``.

## 4. Knowledge graph completion
1.  Preprocess of the data in triplet format:
- ``data prep.ipynb`` in ``code`` folder.
-  Output in ``data``. ``train_neg_tt.zip`` and ``val_neg_tt.zip`` contain the positive and negative triplets from AD. The HD data is used to predict: ``test_drugs.csv`` and ``test_iron_reduced.zip``
   
3. Fine-tuning of the model:
- script in ``code`` named ``biobert_training_weighted_f.py``
- it needs to be executed on a gpu, installing all correct dependencies, to do so run script ``slurm-script.slurm`` in ``code``.
- results of hyperparamter tuning are in ``results``. For each test _x_ there are 4 files: ``epoch_loss_x.txt`` (loss at each iteration of the 3 epochs), ``eval_loss_x.txt`` (loss at each iteration of the validation run), ``map_rel_x.txt`` (dictionary mapping each relationship/class to a numerical label) and ``val_preds_x_i.csv`` (with i=1..5 the predictions of that experiment for the validation set).
- the results of the hyperparameter tuning are analyzed using python methods in ``review.ipynb`` (in ``code``)

4. Predictions:
- script in ``code`` named ``biobert_prediction.py``
