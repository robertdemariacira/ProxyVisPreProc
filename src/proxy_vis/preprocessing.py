from typing import Dict, List

import numpy as np

from proxy_vis import combine_dn_pvis as cmb


def sanitize_data(data_dict: cmb.DataDict, in_place: bool = True) -> cmb.DataDict:
    dd = data_dict
    if not in_place:
        dd = data_dict.copy()
        dd[cmb.BT_TEMP_KEY] = dd[cmb.BT_TEMP_KEY].copy()
        dd[cmb.RADIANCES_KEY] = dd[cmb.RADIANCES_KEY].copy()

    bt = dd[cmb.BT_TEMP_KEY]
    _clamp_sub_dict(bt, in_place)

    rad = dd[cmb.RADIANCES_KEY]
    _clamp_sub_dict(rad, in_place)

    return dd


def _clamp_sub_dict(sub_dict: Dict[str, np.ndarray], in_place: bool) -> None:
    for channel_name, data in sub_dict.items():
        new_data = clamp_to_zero(data, in_place)
        sub_dict[channel_name] = new_data


def sanitize_navigation_arrays(
    navigation: List[np.ndarray], in_place: bool = True
) -> List[np.ndarray]:
    nav_list = navigation
    if not in_place:
        nav_list = list(navigation)

    for i, nav in enumerate(nav_list):
        nav_list[i] = sanitize_navigation(nav, in_place)

    return nav_list


def sanitize_navigation(navigation: np.ndarray, in_place: bool = True) -> np.ndarray:
    invalid_mask = ~np.isfinite(navigation)

    to_alter = navigation
    if not in_place:
        to_alter = np.copy(navigation)

    to_alter[invalid_mask] = np.nan
    return to_alter


def clamp_to_zero(data: np.ndarray, in_place: bool = True) -> np.ndarray:
    to_alter = data
    if not in_place:
        to_alter = np.copy(to_alter)
    to_alter[to_alter < 0] = 0

    return to_alter
