# Data (not included in this repository)

All datasets are public. Download them into the folders below; the scripts expect exactly these file names.

## `data/thiers/` — Lycée Thiers, Marseille, 2013 (SocioPatterns)
Source: http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/
Reference: R. Mastrandrea, J. Fournet, A. Barrat, *PLoS ONE* 10(9): e0136497 (2015).

| Download link on the page | Save as |
|---|---|
| Directed, unweighted friendship network from surveys | `Friendship-network_data_2013.csv.gz` |
| Directed, weighted contact network from contact diaries | `Contact-diaries-network_data_2013.csv.gz` |
| Metadata | `metadata_2013.txt` |

## `data/lazega/` — Lazega law firm (SG&R, 1988–1991)
Source: https://www.stats.ox.ac.uk/~snijders/siena/Lazega_lawyers_data.htm ("Download the data set").
Unzip `LazegaLawyers.zip` here; the scripts use `ELattr.dat`, `ELadv.dat`, `ELfriend.dat`.
Reference: E. Lazega, *The Collegial Phenomenon*, Oxford University Press (2001).

## `data/openalex/` — OpenAlex works
Run `python scripts/04_openalex_download.py` (requires a free OpenAlex API key:
`export OPENALEX_API_KEY=...`). It writes `oa_topics.json` and `oa_works.jsonl.gz` here.
Note: OpenAlex is a live database, so counts will drift slightly from those reported in the paper
(snapshot downloaded 6 November 2026; the scientometrics topic was capped at 40,000 works).
