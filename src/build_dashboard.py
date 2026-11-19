"""Embed out/dashboard_data.json into dashboard/template.html -> dashboard/index.html (self-contained)."""
import json
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = open(os.path.join(root, "out", "dashboard_data.json"), encoding="utf-8").read()
tpl = open(os.path.join(root, "dashboard", "template.html"), encoding="utf-8").read()
assert "/*__DATA__*/null" in tpl
html = tpl.replace("/*__DATA__*/null", data.replace("</", "<\\/"))
out = os.path.join(root, "dashboard", "index.html")
open(out, "w", encoding="utf-8").write(html)
print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
