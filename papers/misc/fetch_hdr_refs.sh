#!/usr/bin/env bash
# fetch_hdr_refs.sh — selection-effects reference PDFs for HDR
# (docs/hubble_diagram_reconstruction.md §5 "Literature"). Written 2026-07-29;
# Claude's sandbox cannot reach arxiv.org (allowlist), so run this yourself once:
#   bash papers/misc/fetch_hdr_refs.sh
set -u
cd "$(dirname "$0")"

fetch () { # $1 = output name, $2 = arxiv path
    if [ ! -s "$1" ]; then
        curl -sL --fail -o "$1" "https://arxiv.org/pdf/$2" \
            && echo "ok   $1" || echo "FAIL $1"
        sleep 1
    else
        echo "have $1"
    fi
}

fetch 1809.02063.pdf      1809.02063        # Mandel, Farr & Gair 2019, MNRAS 486, 1086
fetch 1811.11723.pdf      1811.11723        # Mortlock et al. 2019, PRD 100, 103523
fetch astro-ph_0409387.pdf astro-ph/0409387 # Loredo 2004
fetch 1605.09398.pdf      1605.09398        # Dai, Venumadhav & Sigurdson 2017, PRD 95, 044011
fetch 1807.02584.pdf      1807.02584        # Oguri 2018, MNRAS 480, 3842
fetch 2011.15109.pdf      2011.15109        # Cusin & Tamanini 2021, MNRAS 504, 3610
fetch 2310.12764.pdf      2310.12764        # Canevarolo & Chisari 2024, MNRAS 533, 36
fetch 2402.19476.pdf      2402.19476        # Mpetha et al. 2024, PRD 110, 023502
