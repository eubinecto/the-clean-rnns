from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return:
    """
    # TODO: implement cleansing
    return df


def stratified_split(df: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    stratified-split the given df into two df's.
    """
    total = len(df)
    ratio_size = int(total * ratio)
    other_size = total - ratio_size
    ratio_df, other_df = train_test_split(df, train_size=ratio_size,
                                          stratify=df['label'],
                                          test_size=other_size, random_state=seed,
                                          shuffle=True)
    return ratio_df, other_df

