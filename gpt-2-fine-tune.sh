#!/bin/bash

# cut -d ',' -f 2 haiku.csv > raw-haiku.txt
# shuf raw-haiku.txt > shuf-raw-haiku.txt
# head -n 44295 shuf-raw-haiku.txt > train-raw-haiku.txt
# tail -n 11073 shuf-raw-haiku.txt > test-raw-haiku.txt

## TODO: There are *lots* of warnings.
python3 haikulib/scripts/run_lm_finetuning.py --output_dir=output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=data/train-raw-haiku.txt --do_eval --eval_data_file=data/test-raw-haiku.txt --no_cuda

## TODO: Consider using https://github.com/nshepperd/gpt-2
## TODO: Consider using https://github.com/minimaxir/gpt-2-simple (requires tensorflow <= 1.14)
