def _save_data(adata: AnnData, *, attr: str, key: str, data: Any, prefix: bool = True) -> None:
    obj = getattr(adata, attr)
    obj[key] = data

    if prefix:
        print(f"Adding `adata.{attr}[{key!r}]`")
    else:
        print(f"       `adata.{attr}[{key!r}]`")