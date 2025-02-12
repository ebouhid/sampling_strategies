# This script runs 4 orderings of the 3 evaluated criteria
# for both Landsat-8 and Sentinel.

### LSAT-8 ###
python mlp_curriculum.py --segs_csv dataset_har/lsat_8/dtb.csv \
    --segs_dir dataset_har/lsat_8/ \
    --output results/lsat_8_dtb.csv > results/lsat_8_dtb.out #&

python mlp_curriculum.py --segs_csv dataset_har/lsat_8/entropy.csv \
    --segs_dir dataset_har/lsat_8/ \
    --output results/lsat_8_entropy.csv > results/lsat_8_entropy.out #&

python mlp_curriculum.py --segs_csv dataset_har/lsat_8/time.csv \
    --segs_dir dataset_har/lsat_8/ \
    --output results/lsat_8_time.csv > results/lsat_8_time.out #&


### SENTINEL ###
python mlp_curriculum.py --segs_csv dataset_har/sentinel/dtb.csv \
    --segs_dir dataset_har/sentinel/ \
    --output results/sentinel_dtb.csv > results/sentinel_dtb.out #&

python mlp_curriculum.py --segs_csv dataset_har/sentinel/entropy.csv \
    --segs_dir dataset_har/sentinel/ \
    --output results/sentinel_entropy.csv > results/sentinel_entropy.out #&

python mlp_curriculum.py --segs_csv dataset_har/sentinel/time.csv \
    --segs_dir dataset_har/sentinel/ \
    --output results/sentinel_time.csv > results/sentinel_time.out #&

echo "Done!"
