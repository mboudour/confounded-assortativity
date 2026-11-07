# Download network-science works (2015-2024) from OpenAlex.
# Output: data/openalex/oa_topics.json, data/openalex/oa_works.jsonl.gz
import json, gzip, os, time, urllib.request, urllib.parse

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "openalex"); os.makedirs(OUT, exist_ok=True)
BASE = "https://api.openalex.org"
KEY = os.environ.get("OPENALEX_API_KEY", "")          # optional
QUERIES = ["complex network", "opinion dynamics", "social network analysis",
           "epidemic spreading network", "community detection network"]
YEARS = "2015-2024"
CAP = 40000                                           # max works per topic
FIELDS = "id,publication_year,primary_topic,authorships,referenced_works"

def get(path, **p):
    if KEY: p["api_key"] = KEY
    url = f"{BASE}/{path}?{urllib.parse.urlencode(p)}"
    for k in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print("  retry", k + 1, e); time.sleep(2 ** k)
    raise SystemExit("failed: " + url)

topics = {}
for q in QUERIES:
    t = get("topics", search=q, per_page=1)["results"][0]
    topics[t["id"].split("/")[-1]] = t["display_name"]
    print(f"{q!r:32} -> {t['id'].split('/')[-1]} {t['display_name']}")
json.dump(topics, open(os.path.join(OUT, "oa_topics.json"), "w"), indent=1)

with gzip.open(os.path.join(OUT, "oa_works.jsonl.gz"), "wt") as out:
    for tid, name in topics.items():
        flt = f"primary_topic.id:{tid},publication_year:{YEARS},type:article"
        cursor, n = "*", 0
        total = get("works", filter=flt, per_page=1)["meta"]["count"]
        print(f"{tid} {name}: {total} works (cap {CAP})")
        while cursor and n < CAP:
            d = get("works", filter=flt, per_page=200, cursor=cursor, select=FIELDS)
            for w in d["results"]:
                out.write(json.dumps(w) + "\n"); n += 1
            cursor = d["meta"].get("next_cursor")
            print(f"  {n}", end="\r"); time.sleep(0.12)
        print(f"  done {n}")
print("Finished.")
