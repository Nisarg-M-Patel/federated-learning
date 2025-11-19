ALPHAS=(100 1)
CLIENT_PARTICIPATION_RATIOS=(0.5 0.4)


LOG_DIR=$1
if [[ -z ${LOG_DIR} ]]; then
    echo "[x] ERROR: Please provide a directory to output results to as the first argument."
    exit 2
fi

execute_experiment () {
    LOG_DIR=$1
    LOG_BASE=$2

    if [ ! -f "${LOG_DIR}/${LOG_BASE}.log" ]; then
        if ! python3 main.py --client_participation_ratio ${CLIENT_RATIO} --alpha ${ALPHA} --partition hetero --data-dir ../data > ${LOG_DIR}/${LOG_BASE}.log; then
            echo "[x] Failed in the execution of ${LOG_BASE}. Exiting."
            exit 3
        fi
    else
        echo "[x] ${LOG_DIR}/${LOG_BASE}/${LOG_BASE}.log already exists."
    fi
    echo "[x] Finished execution of ${LOG_BASE}."
}

for ALPHA in ${ALPHAS[@]}; do
    for CLIENT_RATIO in ${CLIENT_PARTICIPATION_RATIOS[@]}; do
        LOG_BASE=client_pariticpation_${CLIENT_RATIO}_noniidness_${ALPHA}
        execute_experiment ${LOG_DIR} ${LOG_BASE}
    done
done