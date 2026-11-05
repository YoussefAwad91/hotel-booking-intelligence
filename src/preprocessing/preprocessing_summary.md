# Preprocessing Specification (Hotel Bookings, target-agnostic)

Script: `preprocess.py` | Run:
`python preprocess.py --train train.csv --val val.csv --test test.csv --out_dir processed`

Optional flags: `--train_out/--val_out/--test_out <path>` (override each output file), `--preprocessor_out <path>`, `--log_cols <cols...>`, `--cyclical`, `--parquet`.

**Golden rule:** every statistic is computed on the **training file only**, stored in the fitted `HotelPreprocessor` (`preprocessor.pkl`) and re-applied unchanged to validation and test. Nothing from val/test is used to fit anything.

**Target-agnostic:** no column is treated as a target. All columns (including `IsCanceled`) are processed as features so any of the 18 models can pick its own target afterwards. Nothing is dropped, no rows are removed, no duplicates are removed.

## 1. Missing values
| Column(s) | Rule |
|---|---|
| `Children` | `NA` -> `0` (EDA: the only column with true missing data) |
| `Country`, `Agent`, `Company` | `nan` / `NULL` / blank -> own category **`Undefined`** (missing % left as is) |
| everything else | No imputation. If any unexpected missing/non-numeric value appears, the script **stops with an error** instead of guessing |

Note: the literal text `NA` in `Country` is kept as its own country (Namibia), not treated as missing.

## 2. Dates (all original date columns kept)
| Column | Rule |
|---|---|
| `ArrivalDateYear`, `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | Kept as numeric, scaled to [0,1] (Year has 3 values -> 0, 0.5, 1; scaled for loss normalization) |
| `ArrivalDateMonth` | Month name -> number 1..12, scaled |
| `ReservationStatusDate` | Parsed `dd/mm/yy` (fallbacks `dd/mm/yyyy`, `yyyy-mm-dd`) -> days since 1970-01-01, scaled. Unparsable value -> error |

### Optional cyclical features (`--cyclical`, OFF by default)
Month and week number are circular (December is next to January) but numeric coding puts them at opposite ends (12 vs 1). With the flag, two columns per feature are added **in addition to** the plain numeric one: `X_sin`, `X_cos` = `(sin(2*pi*X/P)+1)/2` and `(cos(2*pi*X/P)+1)/2`, with period P = 12 (month) and 53 (week number). Together they place each value on a circle so neighbours stay close. No fitting is needed, so there is no leakage. Mainly helps linear models and neural nets; tree models can split on the plain number.

## 3. Categorical encoding
| Features | Encoding |
|---|---|
| Meal, MarketSegment, DistributionChannel, ReservedRoomType, AssignedRoomType, DepositType, CustomerType, ReservationStatus | **One-hot** (0/1), categories learned on train, no level dropped. Names: `Col_Category`. Unseen category in val/test -> all zeros. **Not scaled** (already 0/1) |
| Country, Agent, Company | **Frequency**: relative frequency of the category in train (`count/N`) -> `Col_freq`. Unseen -> 0. `Undefined` is its own category (so `Company_freq` for missing is about 0.95) |

## 4. Numerical scaling to [0, 1]
`x' = (x - min_train) / (max_train - min_train)` (float64, full precision in the CSV).

- Scaled: all numeric columns (LeadTime, ArrivalDate parts, stays, guests, previous bookings/cancellations, BookingChanges, DaysInWaitingList, ADR, parking, special requests, IsRepeatedGuest, IsCanceled, ReservationStatusDate) and the three `_freq` columns.
- `_freq` columns use `min = 0` (not the observed min), so an unseen category is exactly 0.
- Constant train column -> 0.
- **No clipping.** Val/test values beyond the train min/max land slightly outside [0,1] (the script reports how many). Clipping can still be done afterwards if a model needs it; un-clipping is impossible.

**Why [0,1] rather than [-1,1]:** it matches the one-hot and frequency columns already in [0,1]; keeps zeros as zeros (sparse count-like columns such as Babies, PreviousCancellations); suits ReLU/sigmoid nets; trees are indifferent. `[-1,1]` is just `2x'-1` if a tanh-based model ever wants it.

## 5. Optional log transform (`--log_cols`, OFF by default)
For heavy-tailed columns, plain min-max squeezes most rows near 0. Option: `--log_cols LeadTime ADR ...` applies `sign(x)*log(1+|x|)` (works for 0 and negatives, e.g. ADR) **before** min-max. Min/max are then taken from the log-transformed train values. The script prints train skewness for every numeric column and flags `|skew| > 1` as candidates; **nothing is logged unless you list the columns.** Log is monotonic so tree models are unaffected; it mostly helps linear models and neural nets.

## 6. Leakage controls
1. Category lists, frequency maps and min/max come from train only.
2. Date-to-number conversion uses a fixed constant (1970-01-01), no fitting.
3. Cyclical encodings are pure functions of the value, no fitting.
4. Unseen val/test categories get a fixed default (0), and are reported.
5. Row counts are asserted equal before/after for every file and printed with a grand total.

## 7. Outputs
Three CSVs with header row (default `processed/{train,val,test}_processed.csv`, each path overridable) and `preprocessor.pkl`. With `--parquet`, a `.parquet` copy is written next to each CSV (same name, same values; requires `pyarrow`). Column order follows the input file, with each categorical replaced in place by its encoded columns.

## 8. Items for downstream awareness (nothing is dropped here; drop per model later)
`ReservationStatus` and `ReservationStatusDate` reveal whether a booking was canceled, and `AssignedRoomType` is only known at/after check-in. They are kept, but should be excluded for any model where those facts would not be available at prediction time (e.g. predicting `IsCanceled`).
