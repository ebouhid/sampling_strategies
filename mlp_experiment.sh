# Check if lsat.out and senti.out exist and delete them
if [ -f "lsat.out" ]; then
    rm lsat.out
fi
if [ -f "senti.out" ]; then
    rm senti.out
fi

nohup python mlp_curriculum.py --segs_csv dataset_har/sentinel/entropy.csv \
    -e $1 \
    --segs_dir dataset_har/sentinel/ \
    --output results/sentinel_entropy.csv > results/senti.out 2>&1 &

nohup python mlp_curriculum.py --segs_csv dataset_har/lsat_8/entropy.csv \
    -e $1 \
    --segs_dir dataset_har/lsat_8/ \
    --output results/lsat_8_entropy.csv > results/land.out 2>&1 &
