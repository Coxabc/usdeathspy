from pathlib import Path

import polars as pl

from .get_available_years import DATA_TYPES, _search_type


def _load_metadata(obj_name: str) -> pl.DataFrame:
    """Load a single metadata parquet file by object name."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    path = data_dir / f"{obj_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pl.read_parquet(path)


def _is_junk(series: pl.Series) -> pl.Series:
    """Boolean mask – True for rows whose name contains 'filler' or 'reserved'."""
    return series.str.contains(r"(?i)filler|reserved")


def get_fields(
    *meta_frames: pl.DataFrame,
    type: str = "mortality_multiple",
    years: list[int] | None = None,
) -> pl.DataFrame:
    """
    Get and compare available fields across years.

    Returns a summary :class:`polars.DataFrame` of available fields
    (columns) for a given data type and set of years. When multiple years
    are provided the function identifies which fields are present in *all*
    years and which are unique to particular years, making it straightforward
    to understand schema changes across NCHS data releases.

    Parameters
    ----------
    *meta_frames : polars.DataFrame
        One or more metadata DataFrames to inspect directly. Used when
        *years* is ``None``. Each frame must contain at least a ``"name"``
        column and optionally ``"description"`` and ``"codes"`` columns.
    type : str, optional
        The data category to query. Must be one of:

        - ``"mortality_multiple"`` *(default)*
        - ``"births"``
        - ``"birth_cohort"``
        - ``"fetal_deaths"``

        Ignored when explicit *meta_frames* are supplied.
    years : list[int] or None, optional
        Integer years whose metadata files should be loaded automatically.
        When provided, *meta_frames* is ignored and files are resolved from
        the package ``data/`` directory using the pattern
        ``data_<type>_<year>.parquet``.

    Returns
    -------
    polars.DataFrame
        A DataFrame with the following columns:

        ``name``
            Field / variable name as it appears in the raw data file.
        ``description``
            Human-readable label for the field.
        ``has_codes``
            ``"yes"`` if a code lookup table is attached to this field,
            otherwise an empty string.
        ``note``
            Empty string when the field is present in all requested years;
            ``"only in: <years>"`` when the field appears in a subset of
            years only.

    Examples
    --------
    Load fields for a single year:

    >>> from usdeaths import get_fields
    >>> get_fields(type="births", years=[2015])

    Compare fields across multiple years to spot schema changes:

    >>> get_fields(type="births", years=[2003, 2014, 2022])

    Pass pre-loaded DataFrames directly:

    >>> import polars as pl
    >>> meta_2010 = pl.read_parquet("data/data_births_2010.parquet")
    >>> meta_2020 = pl.read_parquet("data/data_births_2020.parquet")
    >>> get_fields(meta_2010, meta_2020)

    Notes
    -----
    Rows whose ``name`` field matches the pattern ``(?i)filler|reserved``
    are treated as padding / reserved fields and excluded from the output.

    The function prints the result to stdout and returns it so it can also
    be captured and used programmatically.

    See Also
    --------
    get_available_years : Discover which years have metadata available.
    """
    if type not in DATA_TYPES:
        raise ValueError(f"'type' must be one of {sorted(DATA_TYPES)}, got {type!r}.")

    # ── Load metadata ────────────────────────────────────────────────────────
    if years is not None:
        st = _search_type(type)
        meta_list = [_load_metadata(f"data_{st}_{yr}") for yr in years]
    else:
        meta_list = list(meta_frames)

    if not meta_list:
        raise ValueError("Provide either explicit metadata frames or a list of years.")

    year_labels = [str(y) for y in years] if years is not None else [str(i) for i in range(len(meta_list))]

    # ── Field presence across years ──────────────────────────────────────────
    field_presence = pl.concat(
        [
            m.filter(~_is_junk(m["name"]))
             .select("name")
             .with_columns(pl.lit(yr).alias("year"))
            for m, yr in zip(meta_list, year_labels)
        ]
    )

    # ── Base result from first metadata frame ────────────────────────────────
    available_cols = [c for c in ("name", "description", "codes") if c in meta_list[0].columns]
    result = (
        meta_list[0]
        .filter(~_is_junk(meta_list[0]["name"]))
        .select(available_cols)
    )

    all_names = field_presence["name"].unique().to_list()
    missing_from_first = set(all_names) - set(result["name"].to_list())

    # ── Pull fields missing from the first frame out of later frames ─────────
    if missing_from_first:
        extras = []
        for m in meta_list[1:]:
            available = [c for c in ("name", "description", "codes") if c in m.columns]
            chunk = (
                m.filter(
                    m["name"].is_in(list(missing_from_first)) & ~_is_junk(m["name"])
                )
                .select(available)
            )
            extras.append(chunk)

        if extras:
            extra_df = (
                pl.concat(extras, how="diagonal")
                .unique(subset=["name"], keep="first")
            )
            result = pl.concat([result, extra_df], how="diagonal")

    # ── Year coverage summary ────────────────────────────────────────────────
    year_coverage = (
        field_presence
        .group_by("name")
        .agg(
            pl.col("year").sort().str.join(", ").alias("available_years"),
            (pl.count() == len(meta_list)).alias("in_all_years"),
        )
    )

    # ── Join and compute derived columns ─────────────────────────────────────
    result = (
        result
        .join(year_coverage, on="name", how="left")
        .with_columns(
            pl.when(pl.col("codes").cast(pl.Utf8).str.len_chars() > 0)
              .then(pl.lit("yes"))
              .otherwise(pl.lit(""))
              .alias("has_codes"),

            pl.when(pl.lit(len(meta_list) == 1))
              .then(pl.lit(""))
              .when(~pl.col("in_all_years"))
              .then(pl.lit("only in: ") + pl.col("available_years"))
              .otherwise(pl.lit(""))
              .alias("note"),
        )
        .select("name", "description", "has_codes", "note")
    )

    print(result)
    return result