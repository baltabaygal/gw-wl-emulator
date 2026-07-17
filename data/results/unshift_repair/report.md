# Post-hoc un-shift repair of the batch-anchor contamination (2026-07-13)

Every len//8 shard of every cached array un-shifted by its own monster-ray anchor estimate delta = -2*sum(kappa>=3)/n (rays dropped, batch shifted back); floors, JSDs, excesses and block nulls recomputed on the repaired arrays with the ORIGINAL bin edges. Repair emulates the robust-anchor fix to first order; study caches untouched. See module docstring for limitations.

## Repaired shards (|delta| > 1e-4 shown)

| array | shard | n | n_monster | kappa_max | delta |
|:--|--:|--:|--:|--:|--:|
| mmin_pd_convergence/z5_main_truthB | 4 | 14991 | 1 | 1015 | -0.1354 |
| mmin_convergence/z10_main_nm200ctl | 3 | 29963 | 3 | 518 | -0.0353 |
| mmin_pd_convergence/z1_main_truthB | 7 | 14999 | 1 | 163 | -0.0217 |
| mmin_pd_convergence/z5_main_m1e5 | 7 | 29983 | 3 | 104 | -0.0079 |
| mmin_convergence/z5_main_m1e5 | 5 | 29982 | 1 | 102 | -0.0068 |
| nz_convergence/z10_main_truthA | 1 | 14980 | 3 | 41 | -0.0063 |
| mmin_pd_convergence/z1_main_truthA | 0 | 14999 | 1 | 43 | -0.0057 |
| mmin_pd_convergence/z1_main_m1e5 | 1 | 29999 | 4 | 62 | -0.0056 |
| mmin_pd_convergence/z5_main_m1e6 | 1 | 29981 | 4 | 53 | -0.0049 |
| mmin_convergence/z10_main_truthA | 1 | 14981 | 2 | 22 | -0.0048 |
| mmin_convergence/z5_main_m1e5 | 7 | 29982 | 5 | 44 | -0.0046 |
| mmin_convergence/z5_main_m1e7 | 1 | 29984 | 1 | 64 | -0.0043 |
| mmin_convergence/z5_main_truthA | 5 | 14992 | 3 | 12 | -0.0038 |
| mmin_convergence/z10_main_m1e5 | 4 | 29959 | 5 | 23 | -0.0035 |
| nz_convergence/z5_main_truthA | 5 | 14991 | 2 | 21 | -0.0034 |
| mmin_pd_convergence/z10_main_truthA | 5 | 14980 | 2 | 16 | -0.0033 |
| mmin_convergence/z10_main_truthA | 5 | 14981 | 2 | 18 | -0.0033 |
| mmin_convergence/z10_main_m1e5 | 6 | 29959 | 4 | 34 | -0.0032 |
| mmin_pd_convergence/z1_main_truthB | 3 | 14999 | 1 | 23 | -0.0031 |
| mmin_pd_convergence/z5_main_m1e5 | 3 | 29983 | 3 | 33 | -0.0029 |
| mmin_pd_convergence/z5_main_truthA | 6 | 14991 | 2 | 11 | -0.0028 |
| mmin_convergence/z5_main_truthB | 6 | 14992 | 2 | 18 | -0.0028 |
| mmin_convergence/z1_sub_truthB | 6 | 15000 | 1 | 20 | -0.0027 |
| mmin_pd_convergence/z1_main_truthB | 6 | 14999 | 3 | 13 | -0.0027 |
| mmin_pd_convergence/z5_main_m1e7 | 5 | 29982 | 1 | 40 | -0.0027 |
| mmin_pd_convergence/z10_main_truthA | 0 | 14980 | 3 | 8 | -0.0025 |
| nz_convergence/z10_main_truthA | 3 | 14980 | 2 | 15 | -0.0025 |
| mmin_pd_convergence/z5_main_truthA | 1 | 14991 | 4 | 5 | -0.0023 |
| nz_convergence/z10_main_truthB | 2 | 14976 | 3 | 8 | -0.0022 |
| mmin_convergence/z10_main_nm200ctl | 6 | 29963 | 1 | 33 | -0.0022 |
| nz_convergence/z5_main_truthA | 0 | 14991 | 3 | 7 | -0.0022 |
| mmin_convergence/z5_main_m1e5 | 6 | 29982 | 1 | 32 | -0.0021 |
| mmin_convergence/z1_main_nm200ctl | 1 | 29999 | 4 | 13 | -0.0021 |
| nz_convergence/z1_main_nz50 | 2 | 29999 | 1 | 31 | -0.0020 |
| mmin_convergence/z10_main_m1e5 | 3 | 29959 | 3 | 16 | -0.0020 |
| mmin_convergence/z5_main_truthB | 3 | 14992 | 3 | 6 | -0.0020 |
| mmin_pd_convergence/z1_main_truthA | 1 | 14999 | 2 | 11 | -0.0020 |
| mmin_pd_convergence/z5_main_truthA | 3 | 14991 | 2 | 11 | -0.0019 |
| mmin_convergence/z10_main_m1e5 | 1 | 29959 | 5 | 7 | -0.0019 |
| nz_convergence/z10_main_nz100 | 7 | 29968 | 1 | 28 | -0.0019 |
| mmin_convergence/z10_main_nm200ctl | 7 | 29963 | 3 | 17 | -0.0018 |
| mmin_convergence/z1_main_truthA | 1 | 14999 | 2 | 9 | -0.0018 |
| nz_convergence/z10_main_truthA | 6 | 14980 | 1 | 13 | -0.0018 |
| mmin_pd_convergence/z5_main_truthB | 0 | 14991 | 2 | 10 | -0.0017 |
| mmin_convergence/z10_main_truthB | 3 | 14979 | 2 | 9 | -0.0017 |
| mmin_pd_convergence/z1_main_truthA | 4 | 14999 | 2 | 7 | -0.0016 |
| mmin_convergence/z1_main_truthB | 2 | 15000 | 1 | 12 | -0.0015 |
| nz_convergence/z5_main_truthB | 1 | 14989 | 2 | 6 | -0.0015 |
| nz_convergence/z10_main_nz200 | 5 | 29956 | 4 | 10 | -0.0015 |
| mmin_convergence/z10_main_m1e6 | 5 | 29961 | 3 | 14 | -0.0014 |
| mmin_convergence/z5_main_m1e6 | 2 | 29982 | 3 | 8 | -0.0013 |
| mmin_pd_convergence/z10_main_truthB | 0 | 14979 | 2 | 6 | -0.0013 |
| nz_convergence/z5_main_nz200 | 0 | 29978 | 1 | 20 | -0.0013 |
| mmin_convergence/z5_main_truthB | 2 | 14992 | 2 | 6 | -0.0013 |
| mmin_pd_convergence/z10_main_truthB | 2 | 14979 | 1 | 10 | -0.0013 |
| nz_convergence/z5_main_nz50 | 7 | 29985 | 1 | 20 | -0.0013 |
| mmin_convergence/z5_main_truthB | 5 | 14992 | 2 | 5 | -0.0013 |
| nz_convergence/z5_main_nz25 | 3 | 29986 | 2 | 15 | -0.0012 |
| mmin_convergence/z10_main_truthB | 4 | 14979 | 1 | 9 | -0.0012 |
| nz_convergence/z10_main_nz100 | 5 | 29968 | 2 | 11 | -0.0012 |
| mmin_pd_convergence/z1_main_truthB | 4 | 14999 | 1 | 9 | -0.0012 |
| mmin_convergence/z5_main_truthB | 7 | 14992 | 2 | 5 | -0.0012 |
| mmin_convergence/z10_main_truthA | 3 | 14981 | 2 | 5 | -0.0012 |
| mmin_convergence/z1_main_truthB | 6 | 15000 | 2 | 5 | -0.0012 |
| nz_convergence/z10_main_truthA | 4 | 14980 | 2 | 5 | -0.0011 |
| mmin_pd_convergence/z1_main_m1e5 | 6 | 29999 | 1 | 17 | -0.0011 |
| mmin_pd_convergence/z10_main_m1e6 | 2 | 29963 | 2 | 9 | -0.0011 |
| mmin_pd_convergence/z5_main_truthA | 4 | 14991 | 2 | 5 | -0.0011 |
| mmin_pd_convergence/z10_main_m1e6 | 7 | 29963 | 2 | 11 | -0.0011 |
| mmin_convergence/z5_main_truthA | 3 | 14992 | 2 | 4 | -0.0011 |
| mmin_pd_convergence/z10_main_truthB | 4 | 14979 | 1 | 8 | -0.0010 |
| mmin_convergence/z1_main_nm200ctl | 3 | 29999 | 2 | 9 | -0.0010 |
| mmin_convergence/z5_main_nm200ctl | 6 | 29984 | 2 | 10 | -0.0010 |
| mmin_pd_convergence/z5_main_truthA | 5 | 14991 | 2 | 4 | -0.0010 |
| mmin_convergence/z10_main_truthB | 5 | 14979 | 2 | 4 | -0.0010 |
| nz_convergence/z10_main_truthB | 1 | 14976 | 2 | 4 | -0.0010 |
| mmin_convergence/z10_main_truthA | 0 | 14981 | 2 | 4 | -0.0010 |
| mmin_pd_convergence/z5_main_m1e5 | 5 | 29983 | 3 | 6 | -0.0010 |
| mmin_convergence/z10_main_m1e7 | 2 | 29961 | 3 | 7 | -0.0010 |
| mmin_pd_convergence/z10_main_m1e7 | 0 | 29965 | 3 | 6 | -0.0009 |
| mmin_pd_convergence/z1_main_m1e5 | 4 | 29999 | 2 | 8 | -0.0009 |
| mmin_pd_convergence/z10_main_m1e6 | 5 | 29963 | 3 | 5 | -0.0009 |
| mmin_convergence/z1_sub_m1e6_mfloor1e7 | 7 | 29999 | 1 | 14 | -0.0009 |
| mmin_pd_convergence/z10_main_m1e6 | 1 | 29963 | 3 | 5 | -0.0009 |
| mmin_convergence/z5_main_m1e7 | 2 | 29984 | 1 | 13 | -0.0009 |
| nz_convergence/z10_main_truthA | 2 | 14980 | 1 | 6 | -0.0008 |
| mmin_pd_convergence/z5_main_truthB | 7 | 14991 | 1 | 6 | -0.0008 |
| mmin_convergence/z1_main_truthB | 4 | 15000 | 1 | 6 | -0.0008 |
| nz_convergence/z10_main_nz50 | 6 | 29963 | 2 | 9 | -0.0008 |
| mmin_pd_convergence/z10_main_m1e5 | 7 | 29965 | 2 | 7 | -0.0008 |
| mmin_convergence/z10_main_m1e5 | 7 | 29959 | 2 | 9 | -0.0008 |
| mmin_convergence/z5_main_nm200ctl | 2 | 29984 | 2 | 9 | -0.0008 |
| mmin_convergence/z1_sub_m1e6_mfloor1e7 | 5 | 29999 | 2 | 9 | -0.0008 |
| mmin_pd_convergence/z10_main_truthA | 4 | 14980 | 1 | 6 | -0.0008 |
| mmin_convergence/z5_main_nm200ctl | 5 | 29984 | 2 | 7 | -0.0008 |
| mmin_convergence/z1_sub_truthA | 5 | 14999 | 1 | 6 | -0.0008 |
| mmin_convergence/z10_main_truthB | 0 | 14979 | 1 | 6 | -0.0008 |
| mmin_convergence/z5_main_truthB | 4 | 14992 | 1 | 6 | -0.0008 |
| mmin_convergence/z5_main_m1e6 | 6 | 29982 | 1 | 12 | -0.0008 |
| mmin_pd_convergence/z1_main_truthB | 1 | 14999 | 1 | 6 | -0.0008 |
| mmin_convergence/z5_main_truthA | 0 | 14992 | 1 | 6 | -0.0007 |
| mmin_pd_convergence/z5_main_m1e5 | 2 | 29983 | 1 | 11 | -0.0007 |
| mmin_convergence/z5_main_m1e5 | 4 | 29982 | 1 | 11 | -0.0007 |
| nz_convergence/z10_main_nz50 | 7 | 29963 | 1 | 11 | -0.0007 |
| mmin_convergence/z10_main_m1e6 | 7 | 29961 | 1 | 11 | -0.0007 |
| mmin_convergence/z5_main_m1e5 | 3 | 29982 | 2 | 7 | -0.0007 |
| mmin_pd_convergence/z10_main_truthB | 6 | 14979 | 1 | 5 | -0.0007 |
| nz_convergence/z5_main_nz200 | 7 | 29978 | 1 | 10 | -0.0007 |
| mmin_convergence/z1_main_m1e6 | 6 | 29999 | 2 | 6 | -0.0007 |
| mmin_pd_convergence/z10_main_m1e6 | 0 | 29963 | 2 | 5 | -0.0007 |
| mmin_convergence/z5_main_m1e5 | 2 | 29982 | 2 | 6 | -0.0007 |
| mmin_convergence/z1_main_nm200ctl | 4 | 29999 | 2 | 5 | -0.0007 |
| mmin_pd_convergence/z1_main_truthA | 7 | 14999 | 1 | 5 | -0.0006 |
| mmin_convergence/z1_main_m1e5 | 6 | 29999 | 2 | 6 | -0.0006 |
| mmin_convergence/z10_main_m1e5 | 0 | 29959 | 2 | 6 | -0.0006 |
| mmin_pd_convergence/z5_main_m1e5 | 0 | 29983 | 2 | 6 | -0.0006 |
| mmin_pd_convergence/z10_main_m1e8 | 5 | 29959 | 2 | 5 | -0.0006 |
| mmin_convergence/z10_main_m1e8 | 1 | 29964 | 1 | 9 | -0.0006 |
| mmin_pd_convergence/z5_main_truthA | 0 | 14991 | 1 | 5 | -0.0006 |
| nz_convergence/z10_main_truthB | 7 | 14976 | 1 | 5 | -0.0006 |
| nz_convergence/z10_main_nz100 | 2 | 29968 | 1 | 9 | -0.0006 |
| nz_convergence/z5_main_truthB | 7 | 14989 | 1 | 4 | -0.0006 |
| nz_convergence/z0.2_main_truthB | 5 | 15000 | 1 | 4 | -0.0006 |
| mmin_convergence/z10_main_nm200ctl | 4 | 29963 | 2 | 5 | -0.0006 |
| mmin_convergence/z10_main_truthA | 7 | 14981 | 1 | 4 | -0.0006 |
| mmin_convergence/z1_main_truthA | 4 | 14999 | 1 | 4 | -0.0006 |
| mmin_pd_convergence/z10_main_m1e5 | 3 | 29965 | 2 | 5 | -0.0006 |
| mmin_convergence/z10_main_m1e7 | 6 | 29961 | 1 | 8 | -0.0006 |
| mmin_convergence/z1_main_nm200ctl | 7 | 29999 | 1 | 8 | -0.0006 |
| mmin_pd_convergence/z1_main_m1e6 | 3 | 29999 | 2 | 5 | -0.0006 |
| nz_convergence/z5_main_truthB | 5 | 14989 | 1 | 4 | -0.0005 |
| mmin_convergence/z1_main_m1e5 | 1 | 29999 | 1 | 8 | -0.0005 |
| mmin_pd_convergence/z10_main_m1e5 | 6 | 29965 | 2 | 5 | -0.0005 |
| mmin_convergence/z1_main_truthB | 7 | 15000 | 1 | 4 | -0.0005 |
| nz_convergence/z1_sub_truthA | 2 | 14999 | 1 | 4 | -0.0005 |
| mmin_convergence/z5_main_truthA | 2 | 14992 | 1 | 4 | -0.0005 |
| mmin_convergence/z10_main_nm200ctl | 1 | 29963 | 1 | 8 | -0.0005 |
| mmin_convergence/z1_main_nm200ctl | 6 | 29999 | 2 | 4 | -0.0005 |
| mmin_pd_convergence/z10_main_truthA | 6 | 14980 | 1 | 4 | -0.0005 |
| mmin_convergence/z10_main_nm200ctl | 5 | 29963 | 2 | 4 | -0.0005 |
| nz_convergence/z1_main_truthA | 2 | 14999 | 1 | 4 | -0.0005 |
| nz_convergence/z10_main_nz50 | 0 | 29963 | 2 | 4 | -0.0005 |
| mmin_convergence/z1_main_m1e5 | 5 | 29999 | 2 | 4 | -0.0005 |
| mmin_pd_convergence/z10_main_m1e8 | 7 | 29959 | 1 | 7 | -0.0005 |
| mmin_pd_convergence/z10_main_m1e7 | 2 | 29965 | 1 | 7 | -0.0005 |
| nz_convergence/z10_main_nz200 | 1 | 29956 | 2 | 4 | -0.0005 |
| mmin_pd_convergence/z5_main_m1e5 | 1 | 29983 | 1 | 7 | -0.0005 |
| nz_convergence/z5_main_truthB | 4 | 14989 | 1 | 4 | -0.0005 |
| nz_convergence/z10_main_truthA | 7 | 14980 | 1 | 4 | -0.0005 |
| nz_convergence/z10_main_truthB | 5 | 14976 | 1 | 4 | -0.0005 |
| nz_convergence/z1_sub_truthA | 0 | 14999 | 1 | 3 | -0.0005 |
| mmin_convergence/z10_main_m1e5 | 5 | 29959 | 2 | 4 | -0.0005 |
| nz_convergence/z5_main_truthB | 2 | 14989 | 1 | 3 | -0.0005 |
| mmin_pd_convergence/z10_main_truthA | 1 | 14980 | 1 | 3 | -0.0005 |
| nz_convergence/z1_main_truthA | 3 | 14999 | 1 | 3 | -0.0005 |
| mmin_pd_convergence/z10_main_truthB | 1 | 14979 | 1 | 3 | -0.0005 |
| mmin_convergence/z10_main_m1e6 | 1 | 29961 | 1 | 7 | -0.0004 |
| mmin_convergence/z10_main_truthB | 7 | 14979 | 1 | 3 | -0.0004 |
| mmin_pd_convergence/z5_main_m1e6 | 2 | 29981 | 2 | 3 | -0.0004 |
| mmin_convergence/z5_main_m1e8 | 7 | 29986 | 1 | 6 | -0.0004 |
| mmin_pd_convergence/z10_main_truthA | 3 | 14980 | 1 | 3 | -0.0004 |
| nz_convergence/z10_main_nz200 | 2 | 29956 | 1 | 6 | -0.0004 |
| nz_convergence/z5_main_nz200 | 4 | 29978 | 1 | 6 | -0.0004 |
| mmin_pd_convergence/z10_main_m1e6 | 6 | 29963 | 1 | 6 | -0.0004 |
| mmin_pd_convergence/z10_main_truthB | 5 | 14979 | 1 | 3 | -0.0004 |
| nz_convergence/z1_sub_nz100 | 0 | 29999 | 1 | 6 | -0.0004 |
| mmin_convergence/z1_main_truthA | 6 | 14999 | 1 | 3 | -0.0004 |
| mmin_convergence/z10_main_truthA | 4 | 14981 | 1 | 3 | -0.0004 |
| mmin_convergence/z1_main_truthB | 3 | 15000 | 1 | 3 | -0.0004 |
| mmin_convergence/z10_main_m1e9 | 2 | 29968 | 1 | 6 | -0.0004 |
| mmin_convergence/z0.2_main_truthA | 0 | 15000 | 1 | 3 | -0.0004 |
| nz_convergence/z1_sub_truthB | 6 | 14999 | 1 | 3 | -0.0004 |
| mmin_pd_convergence/z1_main_m1e6 | 2 | 29999 | 1 | 6 | -0.0004 |
| mmin_convergence/z5_main_m1e6 | 7 | 29982 | 1 | 6 | -0.0004 |
| mmin_convergence/z10_main_m1e7 | 4 | 29961 | 1 | 6 | -0.0004 |
| nz_convergence/z5_main_nz50 | 5 | 29985 | 1 | 6 | -0.0004 |
| mmin_pd_convergence/z10_main_m1e9 | 0 | 29970 | 1 | 5 | -0.0004 |
| nz_convergence/z1_main_nz200 | 1 | 29999 | 1 | 5 | -0.0004 |
| nz_convergence/z5_main_nz25 | 2 | 29986 | 1 | 5 | -0.0004 |
| mmin_pd_convergence/z10_main_m1e5 | 5 | 29965 | 1 | 5 | -0.0004 |
| nz_convergence/z10_main_nz50 | 3 | 29963 | 1 | 5 | -0.0003 |
| mmin_convergence/z5_main_nm200ctl | 3 | 29984 | 1 | 5 | -0.0003 |
| mmin_convergence/z5_main_m1e6 | 1 | 29982 | 1 | 5 | -0.0003 |
| mmin_pd_convergence/z1_main_m1e7 | 3 | 29999 | 1 | 5 | -0.0003 |
| mmin_convergence/z5_main_m1e6 | 4 | 29982 | 1 | 5 | -0.0003 |
| mmin_convergence/z1_main_m1e6 | 4 | 29999 | 1 | 5 | -0.0003 |
| mmin_convergence/z1_main_m1e5 | 3 | 29999 | 1 | 5 | -0.0003 |
| mmin_pd_convergence/z5_main_m1e7 | 4 | 29982 | 1 | 5 | -0.0003 |
| nz_convergence/z10_main_nz100 | 3 | 29968 | 1 | 4 | -0.0003 |
| mmin_convergence/z10_main_m1e8 | 4 | 29964 | 1 | 4 | -0.0003 |
| mmin_convergence/z5_main_nm200ctl | 0 | 29984 | 1 | 4 | -0.0003 |
| mmin_convergence/z5_main_m1e7 | 4 | 29984 | 1 | 4 | -0.0003 |
| mmin_convergence/z1_main_m1e5 | 7 | 29999 | 1 | 4 | -0.0003 |
| mmin_convergence/z1_main_m1e7 | 7 | 29999 | 1 | 4 | -0.0003 |
| mmin_convergence/z5_main_m1e7 | 7 | 29984 | 1 | 4 | -0.0003 |
| mmin_convergence/z5_main_m1e9 | 0 | 29989 | 1 | 4 | -0.0003 |
| mmin_pd_convergence/z5_main_m1e6 | 4 | 29981 | 1 | 4 | -0.0003 |
| mmin_pd_convergence/z10_main_m1e8 | 6 | 29959 | 1 | 4 | -0.0003 |
| mmin_convergence/z1_main_nm200ctl | 2 | 29999 | 1 | 4 | -0.0003 |
| mmin_convergence/z10_main_m1e7 | 0 | 29961 | 1 | 4 | -0.0003 |
| mmin_convergence/z10_main_m1e6 | 0 | 29961 | 1 | 4 | -0.0003 |
| mmin_pd_convergence/z10_main_m1e5 | 0 | 29965 | 1 | 4 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e7 | 4 | 29965 | 1 | 4 | -0.0002 |
| nz_convergence/z10_main_nz200 | 4 | 29956 | 1 | 4 | -0.0002 |
| nz_convergence/z5_main_nz50 | 0 | 29985 | 1 | 4 | -0.0002 |
| mmin_convergence/z0.2_main_m1e5 | 5 | 30000 | 1 | 4 | -0.0002 |
| nz_convergence/z5_main_nz50 | 6 | 29985 | 1 | 4 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e9 | 4 | 29970 | 1 | 4 | -0.0002 |
| mmin_convergence/z10_main_m1e5 | 2 | 29959 | 1 | 4 | -0.0002 |
| mmin_convergence/z10_main_m1e6 | 3 | 29961 | 1 | 4 | -0.0002 |
| mmin_pd_convergence/z1_main_m1e7 | 6 | 29999 | 1 | 4 | -0.0002 |
| mmin_pd_convergence/z5_main_m1e7 | 7 | 29982 | 1 | 4 | -0.0002 |
| mmin_convergence/z5_main_m1e5 | 0 | 29982 | 1 | 4 | -0.0002 |
| mmin_convergence/z5_main_nm200ctl | 1 | 29984 | 1 | 4 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e7 | 5 | 29965 | 1 | 3 | -0.0002 |
| mmin_convergence/z10_main_nm200ctl | 2 | 29963 | 1 | 3 | -0.0002 |
| mmin_convergence/z10_main_m1e9 | 4 | 29968 | 1 | 3 | -0.0002 |
| mmin_convergence/z1_sub_m1e7 | 4 | 29999 | 1 | 3 | -0.0002 |
| mmin_convergence/z1_sub_m1e6_mfloor1e7 | 2 | 29999 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e5 | 1 | 29965 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e5 | 4 | 29965 | 1 | 3 | -0.0002 |
| nz_convergence/z1_main_nz25 | 1 | 29999 | 1 | 3 | -0.0002 |
| nz_convergence/z5_main_nz200 | 3 | 29978 | 1 | 3 | -0.0002 |
| nz_convergence/z5_main_nz200 | 2 | 29978 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z10_main_m1e6 | 4 | 29963 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z5_main_m1e6 | 7 | 29981 | 1 | 3 | -0.0002 |
| nz_convergence/z10_main_nz200 | 0 | 29956 | 1 | 3 | -0.0002 |
| nz_convergence/z5_main_nz100 | 5 | 29984 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z1_main_m1e5 | 2 | 29999 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z5_main_m1e8 | 6 | 29986 | 1 | 3 | -0.0002 |
| mmin_pd_convergence/z5_main_m1e7 | 1 | 29982 | 1 | 3 | -0.0002 |
| nz_convergence/z10_main_nz100 | 4 | 29968 | 1 | 3 | -0.0002 |

(+0 shards with |delta| <= 1e-4 also repaired)

## Repaired floors and excesses

| group | floor_half orig -> rep | block med (rep) | block pctl (rep) | config | excess orig -> rep |
|:--|:--|--:|--:|:--|:--|
| mmin_convergence/z0.2_main | 3.026e-04 -> 2.403e-04 | 2.932e-04 | 13 | m1e5 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e6 | 7.593e-05 -> 6.177e-05 |
| | | | | m1e7 | 4.154e-04 -> 3.438e-04 |
| | | | | m1e8 | 1.005e-03 -> 8.766e-04 |
| | | | | m1e9 | 2.472e-03 -> 2.268e-03 |
| | | | | nm200ctl | 1.433e-05 -> 1.851e-05 |
| mmin_convergence/z10_main | 2.290e-04 -> 2.408e-04 | 2.552e-04 | 26 | m1e5 | 3.520e-05 -> 2.823e-05 |
| | | | | m1e6 | 9.862e-05 -> 7.294e-05 |
| | | | | m1e7 | 2.915e-04 -> 2.491e-04 |
| | | | | m1e8 | 6.915e-04 -> 6.319e-04 |
| | | | | m1e9 | 1.579e-03 -> 1.477e-03 |
| | | | | nm200ctl | 5.785e-04 -> 8.277e-04 |
| mmin_convergence/z1_main | 2.351e-04 -> 2.404e-04 | 2.437e-04 | 47 | m1e5 | 2.433e-05 -> 2.016e-05 |
| | | | | m1e6 | 1.746e-04 -> 1.208e-04 |
| | | | | m1e7 | 5.021e-04 -> 3.886e-04 |
| | | | | m1e8 | 8.725e-04 -> 7.074e-04 |
| | | | | m1e9 | 1.584e-03 -> 1.366e-03 |
| | | | | nm200ctl | 1.623e-04 -> 1.969e-04 |
| mmin_convergence/z1_sub | 2.181e-04 -> 2.282e-04 | 2.203e-04 | 60 | m1e6_mfloor1e7 | 4.265e-05 -> 3.594e-05 |
| | | | | m1e7 | 9.677e-05 -> 6.338e-05 |
| mmin_convergence/z5_main | 2.957e-04 -> 2.917e-04 | 2.605e-04 | 79 | m1e5 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e6 | 5.869e-05 -> 4.707e-05 |
| | | | | m1e7 | 2.874e-04 -> 2.800e-04 |
| | | | | m1e8 | 6.053e-04 -> 5.349e-04 |
| | | | | m1e9 | 1.274e-03 -> 1.177e-03 |
| | | | | nm200ctl | 4.897e-04 -> 4.543e-04 |
| mmin_pd_convergence/z0.2_main | 3.216e-04 -> 3.216e-04 | 2.639e-04 | 87 | m1e5 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e6 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e7 | 3.764e-04 -> 3.764e-04 |
| | | | | m1e8 | 8.127e-04 -> 8.127e-04 |
| | | | | m1e9 | 2.137e-03 -> 2.137e-03 |
| mmin_pd_convergence/z10_main | 2.166e-04 -> 2.164e-04 | 2.504e-04 | 12 | m1e5 | 2.157e-05 -> 1.879e-05 |
| | | | | m1e6 | 4.939e-05 -> 4.161e-05 |
| | | | | m1e7 | 8.750e-05 -> 7.151e-05 |
| | | | | m1e8 | 1.670e-04 -> 1.495e-04 |
| | | | | m1e9 | 5.707e-04 -> 5.251e-04 |
| mmin_pd_convergence/z1_main | 6.753e-04 -> 1.937e-04 | 2.494e-04 | 3 | m1e5 | 6.592e-05 -> 2.884e-05 |
| | | | | m1e6 | 5.527e-04 -> 9.677e-05 |
| | | | | m1e7 | 9.996e-04 -> 2.173e-04 |
| | | | | m1e8 | 1.671e-03 -> 5.369e-04 |
| | | | | m1e9 | 2.060e-03 -> 7.410e-04 |
| mmin_pd_convergence/z5_main | 2.196e-03 -> 2.949e-04 | 2.710e-04 | 74 | m1e5 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e6 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e7 | 0.000e+00 -> 0.000e+00 |
| | | | | m1e8 | 4.293e-04 -> 9.361e-05 |
| | | | | m1e9 | 9.002e-04 -> 2.522e-04 |
| nz_convergence/z0.2_main | 3.277e-04 -> 2.879e-04 | 2.428e-04 | 91 | nz100 | 1.216e-04 -> 6.777e-05 |
| | | | | nz200 | 0.000e+00 -> 0.000e+00 |
| | | | | nz25 | 3.855e-03 -> 3.426e-03 |
| | | | | nz50 | 1.091e-03 -> 8.632e-04 |
| nz_convergence/z10_main | 2.142e-04 -> 2.301e-04 | 2.627e-04 | 13 | nz100 | 1.095e-04 -> 8.376e-05 |
| | | | | nz200 | 9.681e-05 -> 8.103e-05 |
| | | | | nz25 | 5.724e-04 -> 5.060e-04 |
| | | | | nz50 | 1.886e-04 -> 1.582e-04 |
| nz_convergence/z1_main | 2.352e-04 -> 2.390e-04 | 2.486e-04 | 37 | nz100 | 6.458e-05 -> 5.902e-05 |
| | | | | nz200 | 1.248e-05 -> 1.212e-05 |
| | | | | nz25 | 1.468e-03 -> 1.449e-03 |
| | | | | nz50 | 1.726e-04 -> 1.965e-04 |
| nz_convergence/z1_sub | 2.273e-04 -> 2.221e-04 | 2.187e-04 | 55 | nz100 | 4.546e-05 -> 4.461e-05 |
| nz_convergence/z5_main | 2.620e-04 -> 2.524e-04 | 2.465e-04 | 55 | nz100 | 4.735e-05 -> 4.059e-05 |
| | | | | nz200 | 0.000e+00 -> 0.000e+00 |
| | | | | nz25 | 6.558e-04 -> 6.315e-04 |
| | | | | nz50 | 2.170e-04 -> 2.100e-04 |

## Validation (unrepaired recomputation vs summary.json)

| study | group | quantity | summary | recomputed |
|:--|:--|:--|--:|--:|
| mmin_convergence | z0.2_main | floor_half | 3.0261e-04 | 3.0261e-04 |
| mmin_convergence | z0.2_main | m1e5 | 1.1251e-04 | 1.1251e-04 |
| mmin_convergence | z0.2_main | m1e6 | 2.2723e-04 | 2.2723e-04 |
| mmin_convergence | z0.2_main | m1e7 | 5.6669e-04 | 5.6669e-04 |
| mmin_convergence | z0.2_main | m1e8 | 1.1561e-03 | 1.1561e-03 |
| mmin_convergence | z0.2_main | m1e9 | 2.6235e-03 | 2.6235e-03 |
| mmin_convergence | z0.2_main | nm200ctl | 1.6564e-04 | 1.6564e-04 |
| mmin_convergence | z10_main | floor_half | 2.2897e-04 | 2.2897e-04 |
| mmin_convergence | z10_main | m1e5 | 1.4969e-04 | 1.4969e-04 |
| mmin_convergence | z10_main | m1e6 | 2.1310e-04 | 2.1310e-04 |
| mmin_convergence | z10_main | m1e7 | 4.0596e-04 | 4.0596e-04 |
| mmin_convergence | z10_main | m1e8 | 8.0601e-04 | 8.0601e-04 |
| mmin_convergence | z10_main | m1e9 | 1.6931e-03 | 1.6931e-03 |
| mmin_convergence | z10_main | nm200ctl | 6.9302e-04 | 6.9302e-04 |
| mmin_convergence | z1_main | floor_half | 2.3507e-04 | 2.3507e-04 |
| mmin_convergence | z1_main | m1e5 | 1.4187e-04 | 1.4187e-04 |
| mmin_convergence | z1_main | m1e6 | 2.9211e-04 | 2.9211e-04 |
| mmin_convergence | z1_main | m1e7 | 6.1959e-04 | 6.1959e-04 |
| mmin_convergence | z1_main | m1e8 | 9.9004e-04 | 9.9004e-04 |
| mmin_convergence | z1_main | m1e9 | 1.7014e-03 | 1.7014e-03 |
| mmin_convergence | z1_main | nm200ctl | 2.7984e-04 | 2.7984e-04 |
| mmin_convergence | z1_sub | floor_half | 2.1810e-04 | 2.1810e-04 |
| mmin_convergence | z1_sub | m1e7 | 2.0582e-04 | 2.0582e-04 |
| mmin_convergence | z1_sub | m1e6_mfloor1e7 | 1.5170e-04 | 1.5170e-04 |
| mmin_convergence | z5_main | floor_half | 2.9571e-04 | 2.9571e-04 |
| mmin_convergence | z5_main | m1e5 | 1.1919e-04 | 1.1919e-04 |
| mmin_convergence | z5_main | m1e6 | 2.0655e-04 | 2.0655e-04 |
| mmin_convergence | z5_main | m1e7 | 4.3529e-04 | 4.3529e-04 |
| mmin_convergence | z5_main | m1e8 | 7.5318e-04 | 7.5318e-04 |
| mmin_convergence | z5_main | m1e9 | 1.4217e-03 | 1.4217e-03 |
| mmin_convergence | z5_main | nm200ctl | 6.3758e-04 | 6.3758e-04 |
| mmin_pd_convergence | z0.2_main | floor_half | 3.2160e-04 | 3.2160e-04 |
| mmin_pd_convergence | z0.2_main | m1e5 | 1.1528e-04 | 1.1528e-04 |
| mmin_pd_convergence | z0.2_main | m1e6 | 1.4416e-04 | 1.4416e-04 |
| mmin_pd_convergence | z0.2_main | m1e7 | 5.3723e-04 | 5.3723e-04 |
| mmin_pd_convergence | z0.2_main | m1e8 | 9.7354e-04 | 9.7354e-04 |
| mmin_pd_convergence | z0.2_main | m1e9 | 2.2983e-03 | 2.2983e-03 |
| mmin_pd_convergence | z10_main | floor_half | 2.1665e-04 | 2.1665e-04 |
| mmin_pd_convergence | z10_main | m1e5 | 1.2988e-04 | 1.2988e-04 |
| mmin_pd_convergence | z10_main | m1e6 | 1.5771e-04 | 1.5771e-04 |
| mmin_pd_convergence | z10_main | m1e7 | 1.9581e-04 | 1.9581e-04 |
| mmin_pd_convergence | z10_main | m1e8 | 2.7528e-04 | 2.7528e-04 |
| mmin_pd_convergence | z10_main | m1e9 | 6.7901e-04 | 6.7901e-04 |
| mmin_pd_convergence | z1_main | floor_half | 6.7533e-04 | 6.7533e-04 |
| mmin_pd_convergence | z1_main | m1e5 | 4.0358e-04 | 4.0358e-04 |
| mmin_pd_convergence | z1_main | m1e6 | 8.9040e-04 | 8.9040e-04 |
| mmin_pd_convergence | z1_main | m1e7 | 1.3373e-03 | 1.3373e-03 |
| mmin_pd_convergence | z1_main | m1e8 | 2.0083e-03 | 2.0083e-03 |
| mmin_pd_convergence | z1_main | m1e9 | 2.3973e-03 | 2.3973e-03 |
| mmin_pd_convergence | z5_main | floor_half | 2.1962e-03 | 2.1962e-03 |
| mmin_pd_convergence | z5_main | m1e5 | 7.2266e-04 | 7.2266e-04 |
| mmin_pd_convergence | z5_main | m1e6 | 9.5390e-04 | 9.5390e-04 |
| mmin_pd_convergence | z5_main | m1e7 | 1.0770e-03 | 1.0770e-03 |
| mmin_pd_convergence | z5_main | m1e8 | 1.5274e-03 | 1.5274e-03 |
| mmin_pd_convergence | z5_main | m1e9 | 1.9982e-03 | 1.9982e-03 |
| nz_convergence | z0.2_main | floor_half | 3.2774e-04 | 3.2774e-04 |
| nz_convergence | z0.2_main | nz25 | 4.0187e-03 | 4.0187e-03 |
| nz_convergence | z0.2_main | nz50 | 1.2544e-03 | 1.2544e-03 |
| nz_convergence | z0.2_main | nz100 | 2.8550e-04 | 2.8550e-04 |
| nz_convergence | z0.2_main | nz200 | 1.3549e-04 | 1.3549e-04 |
| nz_convergence | z10_main | floor_half | 2.1419e-04 | 2.1419e-04 |
| nz_convergence | z10_main | nz25 | 6.7946e-04 | 6.7946e-04 |
| nz_convergence | z10_main | nz50 | 2.9572e-04 | 2.9572e-04 |
| nz_convergence | z10_main | nz100 | 2.1657e-04 | 2.1657e-04 |
| nz_convergence | z10_main | nz200 | 2.0391e-04 | 2.0391e-04 |
| nz_convergence | z1_main | floor_half | 2.3515e-04 | 2.3515e-04 |
| nz_convergence | z1_main | nz25 | 1.5857e-03 | 1.5857e-03 |
| nz_convergence | z1_main | nz50 | 2.9020e-04 | 2.9020e-04 |
| nz_convergence | z1_main | nz100 | 1.8215e-04 | 1.8215e-04 |
| nz_convergence | z1_main | nz200 | 1.3006e-04 | 1.3006e-04 |
| nz_convergence | z1_sub | floor_half | 2.2727e-04 | 2.2727e-04 |
| nz_convergence | z1_sub | nz100 | 1.5909e-04 | 1.5909e-04 |
| nz_convergence | z5_main | floor_half | 2.6202e-04 | 2.6202e-04 |
| nz_convergence | z5_main | nz25 | 7.8679e-04 | 7.8679e-04 |
| nz_convergence | z5_main | nz50 | 3.4805e-04 | 3.4805e-04 |
| nz_convergence | z5_main | nz100 | 1.7836e-04 | 1.7836e-04 |
| nz_convergence | z5_main | nz200 | 1.0827e-04 | 1.0827e-04 |
