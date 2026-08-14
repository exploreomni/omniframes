# API reference

Generated from the source docstrings. Everything listed here is importable from the top-level
`omniframes` package (`functions` is conventionally imported as `F`).

```python
import omniframes as of
from omniframes import functions as F
```

Private members (`_name`) are omitted; they are not part of the public surface and may change
without notice.

---

## Session

::: omniframes.OmniSession

::: omniframes.session.SessionBuilder

::: omniframes.session.DataFrameReader

---

## DataFrame

::: omniframes.DataFrame

::: omniframes.GroupedData

::: omniframes.io.writers.DataFrameWriter

---

## Column

::: omniframes.Column

---

## Functions (`F`)

::: omniframes.functions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - col
        - lit
        - measure
        - sum
        - avg
        - min
        - max
        - count
        - count_distinct
        - udf

---

## Schema types

::: omniframes.OmniSchema

::: omniframes.OmniField

::: omniframes.OmniDataType

---

## Errors and warnings

Every exception omniframes raises derives from `OmniframesError`; every warning derives from
`OmniframesWarning`. HTTP details never leak above the transport — the one exception is
`TransportError.status`, which carries the status code as an attribute because two endpoints
change their advice based on it.

::: omniframes.OmniframesError

::: omniframes.TransportError

::: omniframes.AuthError

::: omniframes.FeatureFlagError

::: omniframes.ModelPermissionError

::: omniframes.QueryError

::: omniframes.QueryTimeoutError

::: omniframes.CompileError

::: omniframes.errors.OmniframesWarning

::: omniframes.TruncationWarning
