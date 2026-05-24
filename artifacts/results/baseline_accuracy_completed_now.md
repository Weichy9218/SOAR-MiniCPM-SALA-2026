# Baseline Accuracy Completed Chunks

Superseded: full public accuracy is now complete. Use `artifacts/results/baseline_accuracy_full_chunked_summary.txt` for the final 150-sample result (`ori_accuracy=82.53`, `overall_accuracy=100.00`).

Timestamp: 2026-05-22T12:23:30+0800

This is **not** the final 150-sample public accuracy. It summarizes only chunks with both `summary.json` and `predictions.jsonl` already written under `artifacts/results/baseline_accuracy_chunks/`.

## Completed So Far

| metric | value |
|---|---:|
| completed samples | 100 / 150 |
| completed chunks | 10 / 15 |
| partial `ori_accuracy` | 76.00 |
| partial `overall_accuracy` | 95.00 |
| score sum | 76.00 |
| generation duration | 4483.42 s |
| output tokens | 305720 |
| input tokens scored | 4671730 |

## By Task

| task | samples | accuracy | score sum |
|---|---:|---:|---:|
| `mcq` | 30 | 60.00 | 18.00 |
| `niah` | 30 | 100.00 | 30.00 |
| `qa` | 30 | 60.00 | 18.00 |
| `fwe` | 10 | 100.00 | 10.00 |

## By Chunk

| chunk | sample range | samples | accuracy | duration (s) | output tokens |
|---|---:|---:|---:|---:|---:|
| `chunk_0000_0000_0009` | 0-9 | 10 | 60.00 | 663.38 | 61868 |
| `chunk_0001_0010_0019` | 10-19 | 10 | 80.00 | 1314.61 | 110037 |
| `chunk_0002_0020_0029` | 20-29 | 10 | 40.00 | 1460.15 | 112989 |
| `chunk_0003_0030_0039` | 30-39 | 10 | 100.00 | 111.17 | 3423 |
| `chunk_0004_0040_0049` | 40-49 | 10 | 100.00 | 148.08 | 3703 |
| `chunk_0005_0050_0059` | 50-59 | 10 | 100.00 | 256.79 | 3426 |
| `chunk_0006_0060_0069` | 60-69 | 10 | 50.00 | 58.80 | 1022 |
| `chunk_0007_0070_0079` | 70-79 | 10 | 60.00 | 113.62 | 1037 |
| `chunk_0008_0080_0089` | 80-89 | 10 | 70.00 | 222.28 | 997 |
| `chunk_0009_0090_0099` | 90-99 | 10 | 100.00 | 134.54 | 7218 |

## Running / Pending

- `chunk_0010_0100_0109` is running and is not included here.
- Chunks `0011` to `0014` are pending.
- Full public accuracy should only be reported after all 150 samples complete.
