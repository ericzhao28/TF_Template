# Eric's Project Template
This is an open source basic template for Eric's machine learning projects and contracted work.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

## Overview

### Features
This cool template features a number of pretty neat features, including:
* Dummy dataset generation for: flat data, sequential data, doubly-sequential data.
* Skeleton for dataset modules with batteries included (ex: download, preprocessing, sequence padding).
* Sample API/CLI deployment scripts.
* Built-in Neo4j support with Cypher intermediary to Neo4j-driver.
* Tensorflow base model featuring checkpoint saving/restoring, writing to Tensorboard, OOP-style model building.
* Sample sequential encoding layers in TF.
* Sample standard optimization/encoding layers in TF.
* Full-featured logging and unit tests.

### Requirements
* Docker-CE version 17.06.2-ce
* Docker Compose version 1.14.0

### Installation
To install, hop into Docker and install the necessary datasets.
```
docker-compose up --build
cd template
bash access_template.sh
python3 -m template.datasets.aux_set.download
```

### Run Template
First, make sure to hop into docker:
```
docker-compose up --build
cd template
bash access_template.sh
```
Start Neo4J:
```
service neo4j start
```
Preprocess dataset:
```
python3 -m template.datasets.test_set.download
py.test template/tests
```
Start training:
```
python3 -m template.src.main.train
```

### Contribute
I appreciate all contributions. Just make a pull request.
Contributors are listed under `contributors.txt`.

## License
MIT License

Copyright (c) 2017 Eric Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

