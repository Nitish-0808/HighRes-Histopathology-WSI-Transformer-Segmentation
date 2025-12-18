import json, os
from torch.utils.data import DataLoader
from Data.dataset_joint import TilesDatasetJoint

def get_loaders_from_manifests(train_csv, val_csv, stats_json,
                               img_size=352, batch_size=4,
                               num_workers=0, pin_memory=True):
    stats = json.load(open(stats_json))
    mean = stats.get("rgb_mean", [0.5,0.5,0.5])
    std  = stats.get("rgb_std",  [0.5,0.5,0.5])

    train_ds = TilesDatasetJoint(train_csv, mean=mean, std=std, img_size=img_size, is_train=True)
    val_ds   = TilesDatasetJoint(val_csv,   mean=mean, std=std, img_size=img_size, is_train=False)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers,
                          pin_memory=pin_memory, persistent_workers=(num_workers>0))
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers>0))
    return train_ld, val_ld
