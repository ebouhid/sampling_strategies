import numpy as np
from skimage.measure import regionprops
import mahotas
import os
from tqdm import tqdm
import sys
import pandas as pd

def get_hor(segment):
    # flattening segment
    segment = segment.flatten()

    NFP = np.where(segment == 2, 1, 0).sum()
    NP = segment.size
    NNP = NP - NFP

    HoR = max([NFP, NNP]) / NP

    return HoR

def get_major_class(segment):
    segment = segment.astype(np.uint8)
    if np.argmax(np.bincount(segment.flatten())) == 2:
        return "forest"
    elif np.argmax(np.bincount(segment.flatten())) == 1:
        return "nonforest"
    else:
        return "notanalyzed"
    
def evaluate_segment(segment):
    classification = get_major_class(segment)

    if (segment.shape[0] * segment.shape[1] > 70) \
        and (classification in ["forest", "nonforest", "notanalyzed"]):
        return True

    return False

if __name__ == "__main__":
    sensor = 'lsat_8'
    # Create directories
    os.makedirs(f'dataset_har/{sensor}/train', exist_ok=True)
    os.makedirs(f'dataset_har/{sensor}/test', exist_ok=True)

    csv_path = 'lsat_8/segments/output_landsat-8_flip_campaign_response_time_sorted.csv'
    df = pd.read_csv(csv_path)

    metrics = []

    for SA in df['Study_Area'].unique():
        region = f"x{SA :02d}"
        df_region = df[df['Study_Area'] == SA]

        image_path = f'/home/ebneto/datasets/scenes_pca/pca_{region}.npy'
        image = np.load(image_path).astype(np.uint8)

        truth_path = f'/home/ebneto/datasets/truth_masks/truth_{region}.npy'
        truth = np.load(truth_path)

        slic_path = f'lsat_8/slics/slics_bkp_flip_campaign_landsat-8/slic_{region}.npy'
        slic = np.load(slic_path)

        assert truth.shape[:2] == slic.shape[:2]
        assert truth.shape[:2] == image.shape[:2]

        props = regionprops(slic)

        for prop in tqdm(props, desc=f"Processing {region}"):
            minr, minc, maxr, maxc = prop.bbox
            segment_image = image[minr:maxr, minc:maxc, :]
            segment_truth = truth[minr:maxr, minc:maxc]
            segment_class = get_major_class(segment_truth)
            segment_hor = get_hor(segment_truth)
            segment_id = prop.label

            scope = 'train' if segment_id in df_region['Segment_id'].values else 'test'
            if evaluate_segment(segment_truth):
                metrics.append({'region': region, 'segment_id': segment_id, 'class_PRODES': segment_class, 'hor': segment_hor, 'scope': scope})
                
                # Computing Haralick features for the segment
                segment_haralick_ch = np.array([mahotas.features.haralick(segment_image[:, :, channel]) for channel in range(segment_image.shape[2])])

                if scope == 'train':
                    filepath = f'dataset_har/{sensor}/{scope}/{region}_{segment_id}.npy'
                else:
                    filepath = f'dataset_har/{sensor}/{scope}/{region}_{segment_id}-{segment_class}.npy'

                # Saving haralick segment
                np.save(filepath, segment_haralick_ch)


        

    class_info_df = pd.DataFrame.from_records(metrics)
    class_info_df.sort_values(by='segment_id', inplace=True)
    print(class_info_df.head())
    print(class_info_df['class_PRODES'].value_counts())    
    class_info_df.to_csv(f'dataset_har/output_time_{sensor}.csv', index=False)

