#!/usr/bin/bash
# Set some random seeds that will be the same for all experiments
SEEDS=(1234 5678 9101112 13141516 17181920)

# Iterate over datsets
for DATASET in vietnam; do
    mkdir -p logs/$DATASET;
    mkdir -p experiments/$DATASET;
    # Iterate over the graph setups
    for SETUP in head_first head_first-inside_label head_final head_final-inside_label head_final-inside_label-dep_edges head_final-inside_label-dep_edges-dep_labels point_to_root; do
        mkdir -p experiments/$DATASET/$SETUP;
        # Run 5 runs, each with a different random seed to get the variation
        echo "Running $DATASET - $SETUP"
        # for RUN in 1 2 3 4 5; do
        for RUN in 1; do
            i=$(($RUN - 1))
            SEED=${SEEDS[i]}
            OUTDIR=experiments/$DATASET/$SETUP/$RUN;
            mkdir -p experiments/$DATASET/$SETUP/$RUN;
            # If a model is already trained, don't retrain
            if [ -f "$OUTDIR"/test.conllu.pred ]; then
                echo "$DATASET-$SETUP-$RUN already trained"
            else
                mkdir -p logs/$DATASET/$SETUP;
                rm -rf logs/$DATASET/$SETUP/$RUN.out;
                bash ./scripts/run_sentgraph.sh  $DATASET $SETUP $RUN $SEED
            fi
        done;
    done;
done;
