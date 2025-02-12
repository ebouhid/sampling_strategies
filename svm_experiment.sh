# # Check if lsat_2.out and senti_2.out exist and delete them
# if [ -f "lsat_2.out" ]; then
#     rm lsat_2.out
# fi
# if [ -f "senti_2.out" ]; then
#     rm senti_2.out
# fi

nohup python svm_curriculum.py --segs_csv dataset_har/sentinel/entropy.csv \
    -e $1 \
    --segs_dir dataset_har/sentinel/ \
    --output results/sentinel_entropy.csv > results/senti.out 2>&1 &

# nohup python svm_random_curriculum.py --segs_csv dataset_har/sentinel/entropy.csv \
#     -e $1 \
#     --segs_dir dataset_har/sentinel/ \
#     --output results/sentinel_entropy_random.csv > results/senti_random.out 2>&1 &

nohup python svm_curriculum.py --segs_csv dataset_har/lsat_8/entropy.csv \
    -e $1 \
    --segs_dir dataset_har/lsat_8/ \
    --output results/lsat_8_entropy.csv > results/land.out 2>&1 &

# nohup python svm_random_curriculum.py --segs_csv dataset_har/lsat_8/entropy.csv \
#     -e $1 \
#     --segs_dir dataset_har/lsat_8/ \
#     --output results/lsat_8_entropy_random.csv > results/land_random.out 2>&1 &
