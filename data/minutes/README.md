# Copom minutes

`corpus.jsonl` holds the minutes of the meetings of the Copom (Comitê de
Política Monetária, Banco Central do Brasil) that fall within the period of the
snapshot, split into their numbered paragraphs, one JSON object per paragraph.

**Source:** Banco Central do Brasil, open-data dataset "Documentos do Copom"
(<https://dadosabertos.bcb.gov.br/dataset/atas-comunicados-copom>), API
`https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes`.

**License:** the dataset is published under the Open Data Commons Open Database
License (ODbL). This file is a derived database made available under the same
license: <https://opendatacommons.org/licenses/odbl/1-0/>. The code in this
repository is under the MIT license; this data is not.

**How it was made:**

```bash
research-agent minutes download --out data/minutes/raw --since 2023-11-01
research-agent minutes build --raw data/minutes/raw \
  --snapshot data/snapshots/economy-2026-11-12.json --out data/minutes/corpus.jsonl
```

Footnote markers, tables and footnotes are dropped; the text of each numbered
paragraph is kept as published, in Portuguese.
