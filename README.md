# CAKE
ACL 2022: CAKE: A Scalable Commonsense-Aware Framework For Multi-View Knowledge Graph Completion

## Introduction
This is the PyTorch implementation of the [CAKE](https://arxiv.org/pdf/2202.13785.pdf) framework. We propose a novel and scalable Commonsense-Aware Knowledge
Embedding (CAKE) framework to automatically extract commonsense from factual triples with entity concepts. The generated commonsense augments effective self-supervision to
facilitate both high-quality negative sampling and joint commonsense and fact-view link prediction.

## An Overview of the CAKE Framework
![image](https://github.com/ngl567/CAKE/blob/master/CAKE%20framework.png)

## Datasets
We provide four datasets: FB15K_concept, FB15K237_concept, depedia and nell. You can find all the datasets as well as the files corresponding to concept in the folders:
* train.txt: the file containing all the triples for training. Each line is a triple in the format (head entity name, relation name, tail entity name).
* test.txt: the file containing all the triples for testing. Each line is a triple in the format (head entity name, relation name, tail entity name).
* valid.txt: the file containing all the triples for validation. Each line is a triple in the format (head entity name, relation name, tail entity name).
* dom_ent.json: the dictionary file denoting all the entities that belong to each concept, in the format {domain_1_id: \[entity_1_id, entity_2_id, ...\]}.
* ent_dom.json: the dictionary file denoting all the concepts corresponding to each entity, in the format {entity_1_id: \[concept_1_id, concept_2_id, ...\]}.
* rel2dom_h.json: the dictionary file denoting all the head concepts corresponding to each relation, in the format {relation_1_id: \[concept_1_id, concept_2_id, ...\]}.
* rel2dom_t.json: the dictionary file denoting all the tail concepts corresponding to each relation, in the format {relation_1_id: \[concept_1_id, concept_2_id, ...\]}.
* rel2nn.json: rel2dom_h.json: the dictionary file denoting the complex relation property of each relation, in the format {relation_1_id: complex_id}. complex_id: 0: 1-1 relation, 1: 1-N relation, 2: N-1 relation, 3: N-N relation.
* entities.dict: the dictionary file containing all the entities in the dataset. Each line is an entity and its id: (entity_id, entity name).
* relations.dict: the dictionary file containing all the relations in the dataset. Each line is an relation and its id: (relation_id, relation name).

## Train
In order to reproduce the results of CAKE model on the datasets, you can kindly run the following commands:  
**TransE+CAKE:**
```
bash run_cake.sh train TransE FB15k-237_concept 0 domain 512 2 1000 12.0 1.0 0.00005 200000 16
bash run_cake.sh train TransE nell 2 mvlp 256 2 500 8.0 0.5 0.0001 250000 8
bash run_cake.sh train TransE dbpedia 0 all 512 2 1000 24.0 1.0 0.0001 200000 8
```

**RotatE+CAKE:**
```
bash run_cake.sh train RotatE FB15k-237_concept 2 0 512 2 1000 20.0 1.0 0.00005 100000 16 -de
bash run_all.sh train RotatE nell 2 all 256 2 500 8.0 0.5 0.0001 250000 8 -de
bash run_cake.sh train RotatE dbpedia 1 mulp 1024 2 500 24.0 1.0 0.0002 200000 4 -de
```

## Acknowledge
If you use the codes, please cite the following paper:
```
@inproceedings{RPJE19,
  author    = {Guanglin Niu and
               Bo Li and
               Yongfei Zhang and
               Shiliang Pu},
  title     = {CAKE: A Scalable Commonsense-Aware Framework For Multi-View Knowledge Graph Completion},
  booktitle = {ACL},
  year      = {2022}
}
```
