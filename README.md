# kg_adhd

## 1. Gathering data

## 2. Building the Knowledge Graphs
- neo4j latest version 5.16 
- through linux
- necessary plugins (apoc)
- commands:
LOAD CSV WITH HEADERS FROM 'file:///graph_nodes_v2024-02-07.csv' AS row CREATE (:Node { id: row.id, semantic_groups: row.semantic_groups, preflabel: row.preflabel, synonyms: row.synonyms, name: row.name, description: row.description }); 

LOAD CSV WITH HEADERS FROM 'file:///edges_copy.csv' AS row MATCH (subject:Node {id: (row.subject_id)}), (object:Node {id: (row.object_id)}) WITH row, subject, object WHERE NOT row.property_label IS NULL AND trim(row.property_label) <> '' CALL apoc.create.relationship(subject, row.property_label, {}, object) YIELD rel SET rel += { reference_uri: row.reference_uri, reference_supporting_text: row.reference_supporting_text, reference_date: row.reference_date, property_description: row.property_description, property_uri: row.property_uri }; 
