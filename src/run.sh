#! /bin/bash

for beta in 1 1e-1 1e-2 1e-3 1e-4; do
    python main.py --beta ${beta}
done