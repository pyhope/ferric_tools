#!/bin/bash

# Run SQS to get the input file:
# corrdump -l=conf.in -ro -noe -nop -clus -2=5.0; getclus 
# mcsqs -l=conf.in -sig=9 -n=xxx -rc -sd=xxx
# cellcvrt -f -sig=9 < bestsqs.out > conf.sqs
# Then convert conf.sqs into POSCAR format

python3 ../../src/intro_ferric.py -s -rs 20250505
