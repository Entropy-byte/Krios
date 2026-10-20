# Validation against the NASA / EclipseWise catalog, 2021-2030

Reference: F. Espenak, EclipseWise decade tables (times are TD = TT of greatest eclipse).
Ephemeris under test: `analytic` (offline Meeus ELP-2000/82 + VSOP87D truncated).

## Solar eclipses

| Date | Type (NASA / krios) | NASA TD | dt (s) | Magnitude NASA / krios | Central duration NASA / krios |
|---|---|---|---:|---|---|
| 2021-06-10 | Annular / Annular | 10:43:07 | +0.9 | 0.944 / 0.943 | 231 s / 231 s |
| 2021-12-04 | Total / Total | 07:34:38 | +18.2 | 1.037 / 1.037 | 114 s / 115 s |
| 2022-04-30 | Partial / Partial | 20:42:37 | -15.4 | 0.640 / 0.639 | - |
| 2022-10-25 | Partial / Partial | 11:01:20 | -3.4 | 0.862 / 0.862 | - |
| 2023-04-20 | Hybrid / Hybrid | 04:17:56 | +2.0 | 1.013 / 1.013 | 76 s / 76 s |
| 2023-10-14 | Annular / Annular | 18:00:41 | +5.7 | 0.952 / 0.952 | 317 s / 317 s |
| 2024-04-08 | Total / Total | 18:18:29 | +5.1 | 1.057 / 1.057 | 268 s / 268 s |
| 2024-10-02 | Annular / Annular | 18:46:13 | -1.6 | 0.933 / 0.933 | 445 s / 445 s |
| 2025-03-29 | Partial / Partial | 10:48:36 | +13.8 | 0.938 / 0.936 | - |
| 2025-09-21 | Partial / Partial | 19:43:04 | -3.6 | 0.855 / 0.855 | - |
| 2026-02-17 | Annular / Annular | 12:13:06 | +5.6 | 0.963 / 0.963 | 140 s / 140 s |
| 2026-08-12 | Total / Total | 17:47:06 | -9.0 | 1.039 / 1.039 | 138 s / 138 s |
| 2027-02-06 | Annular / Annular | 16:00:48 | +0.0 | 0.928 / 0.928 | 471 s / 471 s |
| 2027-08-02 | Total / Total | 10:07:50 | -1.6 | 1.079 / 1.079 | 383 s / 382 s |
| 2028-01-26 | Annular / Annular | 15:08:59 | +2.9 | 0.921 / 0.921 | 627 s / 627 s |
| 2028-07-22 | Total / Total | 02:56:40 | +6.4 | 1.056 / 1.056 | 310 s / 309 s |
| 2029-01-14 | Partial / Partial | 17:13:48 | -0.1 | 0.871 / 0.871 | - |
| 2029-06-12 | Partial / Partial | 04:06:13 | +5.5 | 0.458 / 0.458 | - |
| 2029-07-11 | Partial / Partial | 15:37:19 | +11.6 | 0.230 / 0.230 | - |
| 2029-12-05 | Partial / Partial | 15:03:58 | +0.2 | 0.891 / 0.890 | - |
| 2030-06-01 | Annular / Annular | 06:29:13 | +0.0 | 0.944 / 0.944 | 321 s / 321 s |
| 2030-11-25 | Total / Total | 06:51:37 | +13.9 | 1.047 / 1.047 | 224 s / 223 s |

Solar: 22/22 eclipses found, all types correct; |dt| median 4.4 s, max 18.2 s.

## Lunar eclipses

| Date | Type (NASA / krios) | NASA TD | dt (s) | Umbral magnitude NASA / krios |
|---|---|---|---:|---|
| 2021-05-26 | Total / Total | 11:19:53 | +4.5 | 1.009 / 1.011 |
| 2021-11-19 | Partial / Partial | 09:04:06 | -5.8 | 0.974 / 0.976 |
| 2022-05-16 | Total / Total | 04:12:42 | +4.5 | 1.414 / 1.416 |
| 2022-11-08 | Total / Total | 11:00:22 | -7.9 | 1.359 / 1.361 |
| 2023-05-05 | Penumbral / Penumbral | 17:24:04 | +12.7 | - / -0.043 |
| 2023-10-28 | Partial / Partial | 20:15:17 | -8.0 | 0.122 / 0.124 |
| 2024-03-25 | Penumbral / Penumbral | 07:14:00 | +2.9 | - / -0.131 |
| 2024-09-18 | Partial / Partial | 02:45:26 | +8.8 | 0.085 / 0.086 |
| 2025-03-14 | Total / Total | 06:59:56 | -6.1 | 1.178 / 1.180 |
| 2025-09-07 | Total / Total | 18:12:58 | -5.8 | 1.362 / 1.363 |
| 2026-03-03 | Total / Total | 11:34:52 | +5.0 | 1.151 / 1.153 |
| 2026-08-28 | Partial / Partial | 04:14:04 | +11.0 | 0.930 / 0.931 |
| 2027-02-20 | Penumbral / Penumbral | 23:14:05 | -8.4 | - / -0.057 |
| 2027-07-18 | Penumbral / Penumbral | 16:04:11 | -4.3 | - / -1.066 |
| 2027-08-17 | Penumbral / Penumbral | 07:14:58 | +4.4 | - / -0.523 |
| 2028-01-12 | Partial / Partial | 04:14:13 | +5.6 | 0.066 / 0.068 |
| 2028-07-06 | Partial / Partial | 18:20:57 | -5.6 | 0.389 / 0.391 |
| 2028-12-31 | Total / Total | 16:53:15 | -6.0 | 1.246 / 1.247 |
| 2029-06-26 | Total / Total | 03:23:22 | +7.3 | 1.844 / 1.845 |
| 2029-12-20 | Total / Total | 22:43:12 | +5.0 | 1.117 / 1.118 |
| 2030-06-15 | Partial / Partial | 18:34:33 | -2.6 | 0.503 / 0.504 |
| 2030-12-09 | Penumbral / Penumbral | 22:28:51 | +0.0 | - / -0.162 |

Lunar: 22/22 eclipses found, all types correct; |dt| median 5.7 s, max 12.7 s.
