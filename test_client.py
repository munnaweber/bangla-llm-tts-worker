"""Send a test job and save any audio.

  export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
  python test_client.py tests/tts.json
"""
import base64
import json
import os
import sys
import time
import urllib.request

API = f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}"
HEADERS = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"}


def call(method, url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers=HEADERS, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


payload = json.load(open(sys.argv[1], encoding="utf-8"))
job = call("POST", f"{API}/run", payload)
while True:
    st = call("GET", f"{API}/status/{job['id']}")
    if st["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    print("status:", st["status"])
    time.sleep(2)

out = st.get("output") or {}
if st["status"] != "COMPLETED" or "error" in out:
    sys.exit(f"{st['status']}: {out.get('error') or st.get('error')}")

if "audio_base64" in out:
    path = f"voice.{out['format']}"
    open(path, "wb").write(base64.b64decode(out.pop("audio_base64")))
    print("saved:", path)
print(json.dumps(out, ensure_ascii=False, indent=2))
print("delayTime ms:", st.get("delayTime"), "| executionTime ms:", st.get("executionTime"))
