# Eric's Project Template
This is an open source basic template for Eric's machine learning projects and contracted work.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

## Overview

### Features
This cool template features a number of pretty neat features, including:
- Nice skeleton for datasets
- Useful utility scripts for preprocessing
- Pretty Tensorflow models with a couple prebuilt model layers
- Generic deployment and program logic scripts

### Requirements
* Docker-CE version 17.06.2-ce
* Docker Compose version 1.14.0

### Installation
To install, hop into Docker and install the necessary datasets.
```
docker-compose up --build
cd template
bash access_template.sh
python3 -m template.datasets.primary_set.download
python3 -m template.datasets.aux_set.download
```

### Usage
First, make sure to hop into docker:
```
docker-compose up --build
cd template
bash access_template.sh
```

To deploy the API with a pretrained model: `python3 -m template.src.main.api`
To deploy the CLI with a pretrained model: `python3 -m template.src.main.cli`
To train the model: `python3 -m template.src.main.train`

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

