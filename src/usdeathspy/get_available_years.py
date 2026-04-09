import re
from pathlib import Path

DATA_TYPES = {"mortality_multiple", "births", "birth_cohort", "fetal_deaths"}


def _search_type(type: str) -> str:
    """Handle the singular/plural mismatch for fetal_deaths → fetal_death."""
    return "fetal_death" if type == "fetal_deaths" else type


def _get_package_data_objects() -> list[str]:
    """
    Return all data-object names available in the package.
    Assumes data files live under a `data/` folder next to this module,
    named like  data_births_1968.parquet  (or .csv / .arrow – adjust glob).
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    return [p.stem for p in data_dir.glob("*.parquet")]


def get_available_years(type: str) -> list[int]:
    """
    Get available years for a data type.

    Queries the package data directory to identify years for which metadata
    objects are available for a specific data category. The discovery is
    based on parquet files named with the pattern
    ``data_<type>_<year>.parquet`` (e.g. ``data_births_1968.parquet``).

    Parameters
    ----------
    type : str
        The data category to query. Must be one of:

        - ``"mortality_multiple"`` – multiple-cause-of-death records
        - ``"births"`` – natality / live-birth records
        - ``"birth_cohort"`` – linked birth / infant-death cohort records
        - ``"fetal_deaths"`` – fetal-death records

    Returns
    -------
    list[int]
        Sorted list of integer years for which metadata is available.


    Examples
    --------
    >>> from usdeaths import get_available_years
    >>> get_available_years("births")
    [1968, 1969, 1970, ..., 2022]

    >>> get_available_years("fetal_deaths")
    [1982, 1983, ..., 2022]

    """
    if type not in DATA_TYPES:
        raise ValueError(
            f"'type' must be one of {sorted(DATA_TYPES)}, got {type!r}."
        )

    all_objects = _get_package_data_objects()
    pattern = rf"^data_{_search_type(type)}_\d{{4}}$"

    matching = [obj for obj in all_objects if re.match(pattern, obj)]

    years = sorted(
        int(re.search(r"\d{4}", obj).group())
        for obj in matching
    )

    if not years:
        raise ValueError(f"No metadata objects found for type {type!r}.")

    return years