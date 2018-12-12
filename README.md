# GPUs Outperform Current HPC and Neuromorphic Solutions in Terms of Speed and Energy When Simulating a Highly-Connected Cortical Model
Paper on using GeNN for simulating cortical models

## Building paper
``make bib``

## Building models
1. Install the version of GeNN (3.2.0 release) included in the sub-module using the instructions at https://github.com/genn-team/genn/
2. Both models are located in sub-folders under ``models`` and can be run on Linux or Mac using the following steps (as discussed in the manuscript, the potjans_microcircuit requires a 4GB GPU and the Morrison Aetson Diesmann model a 12GB GPU):
   1. Use GeNN to generate simulation code with ``genn-buildmodel model.cc``
   2. Build simulation code ``make`` or 
   3. Run simulation code using ``./simulator``
