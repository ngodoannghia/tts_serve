#!/bin/bash
set -euxo pipefail

torch-model-archiver --model-name tts_vntre --version 1.0 --model-file models.py --serialized-file generator.pth --handler tts_handler.py --extra-files attentions.py,commons.py,config.json,convert.txt,duration_model.pth,flow.py,modules.py,numtotext.py,phone_set.json,preprocess_new.py,units.txt -r requirements.txt