"""Зареждане и първоначално почистване на KDD Cup 1999."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_kddcup99


COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type"
]


CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag",
]

ATTACK_CATEGORY_MAP = {

    # Нормален трафик
    "normal": "normal",

    # DoS
    "apache2": "dos",
    "back": "dos",
    "land": "dos",
    "mailbomb": "dos",
    "neptune": "dos",
    "pod": "dos",
    "processtable": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "udpstorm": "dos",

    # Probe
    "ipsweep": "probe",
    "mscan": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "saint": "probe",
    "satan": "probe",

    # R2L
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "httptunnel": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "named": "r2l",
    "phf": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "worm": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",

    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "ps": "u2r",
    "rootkit": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}

TARGET_COLUMNS = [
    "attack_type",
    "attack_category",
]

CONSTANT_COLUMNS = [
    "num_outbound_cmds",
    "is_host_login",
]



def decode_value(value):
    """Преобразува byte стойност в обикновен Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value

def add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Групира конкретните етикети в пет основни класа."""
    df = df.copy()

    df["attack_category"] = df["attack_type"].map(
        ATTACK_CATEGORY_MAP
    )

    unknown_labels = df.loc[
        df["attack_category"].isna(),
        "attack_type"
    ].unique()

    if len(unknown_labels) > 0:
        raise ValueError(
            "Неразпознати етикети: "
            + ", ".join(map(str, unknown_labels))
        )

    return df

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразува всички входни не-категориални колони в числов тип."""
    df = df.copy()

    excluded_columns = CATEGORICAL_COLUMNS + TARGET_COLUMNS

    numeric_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    return df

def remove_exact_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Премахва напълно идентичните редове.

    Запазва първото срещане на всеки ред и нулира индекса.
    """
    return (
        df
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )

def drop_constant_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Премахва характеристиките, които са константни
    в обучаващите данни.
    """
    existing_columns = [
        column
        for column in CONSTANT_COLUMNS
        if column in df.columns
    ]

    return df.drop(columns=existing_columns)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Почиства категориалните колони и етикета на набора от данни.
    """
    df = df.copy()

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .apply(decode_value)
            .astype(str)
            .str.strip()
            .str.lower()
        )

    df["attack_type"] = (
        df["attack_type"]
        .apply(decode_value)
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .str.lower()
    )

    return df


def load_training_data() -> pd.DataFrame:
    """
    Зарежда 10% обучаващ набор KDD Cup 1999 чрез scikit-learn.

    Returns:
        Почистен DataFrame с 41 характеристики и колоната attack_type.
    """
    dataset = fetch_kddcup99(
        percent10=True,
        as_frame=True
    )

    df_train = dataset.frame.copy()

    # scikit-learn именува целевата колона "labels".
    df_train = df_train.rename(columns={"labels": "attack_type"})

    df_train = clean_dataframe(df_train)
    df_train = add_attack_category(df_train)
    

    return df_train


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """
    Зарежда и почиства независимия тестов набор corrected.

    Args:
        file_path: Път до файла corrected или corrected.gz.

    Returns:
        Почистен тестов DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Тестовият файл не е намерен: {file_path}"
        )

    df_test = pd.read_csv(
        file_path,
        names=COLUMN_NAMES,
        header=None,
        compression="infer"
    )

    df_test = clean_dataframe(df_test)
    df_test = add_attack_category(df_test)
    df_test = convert_numeric_columns(df_test)
    df_test = drop_constant_columns(df_test)

    return df_test


def load_data(
    test_file_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Зарежда обучаващия и тестовия набор.

    Args:
        test_file_path: Път до corrected или corrected.gz.

    Returns:
        Tuple във формат (df_train, df_test).
    """
    df_train = load_training_data()
    df_test = load_test_data(test_file_path)

    return df_train, df_test