"""Read-only stdio MCP: papers, metadata, and reproducibility evidence."""
from __future__ import annotations
import json, os, re, sys, urllib.parse, urllib.request, subprocess, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[1]; CORPUS=ROOT/"references"/"seed-papers.json"; TIMEOUT=int(os.getenv("SIMPLE_RESEARCH_TIMEOUT_SECONDS","15"))
TOOLS=[
 {"name":"research_search_papers","description":"Search Crossref papers. Read-only; returns DOI, title, year, venue.","inputSchema":{"type":"object","properties":{"query":{"type":"string","minLength":3},"limit":{"type":"integer","minimum":1,"maximum":20}},"required":["query"]}},
 {"name":"research_seed_library","description":"Return curated FX/causality/backtest papers and use-boundaries.","inputSchema":{"type":"object","properties":{"topic":{"type":"string"}},"additionalProperties":False}},
 {"name":"research_doi_metadata","description":"Fetch normalized Crossref metadata for one DOI. Read-only.","inputSchema":{"type":"object","properties":{"doi":{"type":"string","minLength":6}},"required":["doi"]}},
 {"name":"research_evidence_check","description":"Check a proposed claim against required evidence fields; never certifies profitability or promotion.","inputSchema":{"type":"object","properties":{"claim":{"type":"string"},"doi":{"type":"string"},"method":{"type":"string"},"limitations":{"type":"string"}},"required":["claim","doi","method","limitations"]}}
 ,{"name":"agent_repo_snapshot","description":"Read Git branch, HEAD, status, and changed paths. No writes.","inputSchema":{"type":"object","properties":{},"additionalProperties":False}}
 ,{"name":"agent_diff_scope","description":"Read changed paths and flag data, secrets, generated artifacts, and source ownership-risk paths. No writes.","inputSchema":{"type":"object","properties":{"staged":{"type":"boolean"}},"additionalProperties":False}}
 ,{"name":"agent_find_tests","description":"Find tests mentioning a module or keyword. No writes.","inputSchema":{"type":"object","properties":{"query":{"type":"string","minLength":2}},"required":["query"]}}
 ,{"name":"agent_receipt_summary","description":"Read local test receipts and verify receipt JSON hashes. No writes.","inputSchema":{"type":"object","properties":{"limit":{"type":"integer","minimum":1,"maximum":50}},"additionalProperties":False}}
 ,{"name":"agent_tool_health","description":"Report Python, Git, Node, API-key presence only, and active-hook state. Never reveals secrets.","inputSchema":{"type":"object","properties":{},"additionalProperties":False}}
 ,{"name":"quant_protocol_gate","description":"Evaluate explicit quant-research protocol fields. Advisory only; never certifies profitability.","inputSchema":{"type":"object","properties":{"decision_time_defined":{"type":"boolean"},"target_time_defined":{"type":"boolean"},"chronological_holdout_sealed":{"type":"boolean"},"execution_assumptions_defined":{"type":"boolean"},"matched_baseline_defined":{"type":"boolean"},"costs_defined":{"type":"boolean"}},"required":["decision_time_defined","target_time_defined","chronological_holdout_sealed","execution_assumptions_defined","matched_baseline_defined","costs_defined"]}}
 ,{"name":"physics_density_matrix_check","description":"Check Hermiticity, trace, and eigenvalue positivity of a small real/complex density matrix. Advisory numerical check only.","inputSchema":{"type":"object","properties":{"matrix":{"type":"array","minItems":1},"tolerance":{"type":"number","minimum":0}},"required":["matrix"]}}
 ,{"name":"market_dukascopy_history_url","description":"Build official Dukascopy tick-file URL for one UTC hour. No download.","inputSchema":{"type":"object","properties":{"instrument":{"type":"string","pattern":"^[A-Z0-9/]{3,12}$"},"year":{"type":"integer","minimum":2000},"month":{"type":"integer","minimum":1,"maximum":12},"day":{"type":"integer","minimum":1,"maximum":31},"hour_utc":{"type":"integer","minimum":0,"maximum":23}},"required":["instrument","year","month","day","hour_utc"]}}
 ,{"name":"market_dukascopy_probe","description":"Read HTTP metadata for one Dukascopy tick file using a range request; never stores tick payload.","inputSchema":{"type":"object","properties":{"url":{"type":"string","pattern":"^https://datafeed\\.dukascopy\\.com/"}},"required":["url"]}}
 ,{"name":"market_tradingview_chart_url","description":"Build public TradingView chart URL from provider-qualified symbol. No scraping, login, or data API call.","inputSchema":{"type":"object","properties":{"symbol":{"type":"string","pattern":"^[A-Z0-9_:-]{3,64}$"}},"required":["symbol"]}}
 ,{"name":"market_feed_contract","description":"Validate explicit data-feed provenance fields. Advisory only.","inputSchema":{"type":"object","properties":{"provider":{"type":"string"},"symbol":{"type":"string"},"price_side":{"type":"string","enum":["bid","ask","mid","last","unknown"]},"timezone":{"type":"string"},"retrieved_at_utc":{"type":"string"},"execution_feed_same":{"type":"boolean"}},"required":["provider","symbol","price_side","timezone","retrieved_at_utc","execution_feed_same"]}}
]
for _tool in TOOLS:
 _tool["inputSchema"].setdefault("additionalProperties",False)

def _invalid(message:str)->dict:
 return {"content":[{"type":"text","text":json.dumps({"error":message})}],"isError":True,"_invalid_params":True}

def _matches(schema:dict, value:object, path:str="arguments")->str|None:
 kind=schema.get("type")
 if kind=="object":
  if not isinstance(value,dict): return f"{path} must be an object"
  props=schema.get("properties",{}); required=schema.get("required",[])
  for key in required:
   if key not in value: return f"{path}.{key} is required"
  if schema.get("additionalProperties") is False:
   unknown=set(value)-set(props)
   if unknown: return f"{path} has unknown fields: {sorted(unknown)}"
  for key,item in value.items():
   if key in props:
    error=_matches(props[key],item,f"{path}.{key}")
    if error:return error
  return None
 if kind=="string":
  if not isinstance(value,str):return f"{path} must be a string"
  if len(value)<schema.get("minLength",0) or len(value)>schema.get("maxLength",2**31):return f"{path} has invalid length"
  if "pattern" in schema and not re.fullmatch(schema["pattern"],value):return f"{path} does not match required pattern"
 if kind=="integer":
  if type(value) is not int:return f"{path} must be an integer"
 if kind=="number":
  if type(value) not in (int,float) or isinstance(value,bool):return f"{path} must be a number"
 if kind=="boolean" and type(value) is not bool:return f"{path} must be boolean"
 if kind=="array" and not isinstance(value,list):return f"{path} must be an array"
 if "enum" in schema and value not in schema["enum"]:return f"{path} is not an allowed value"
 if "minimum" in schema and value<schema["minimum"]:return f"{path} is below minimum"
 if "maximum" in schema and value>schema["maximum"]:return f"{path} is above maximum"
 return None

def _tool_schema(name:str)->dict|None:
 return next((tool["inputSchema"] for tool in TOOLS if tool["name"]==name),None)

class _NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): raise urllib.error.HTTPError(args[0], 302, "redirects are not permitted", args[3], None)

def _dukascopy_url(value:str)->str:
 parsed=urllib.parse.urlsplit(value)
 if (parsed.scheme != "https" or parsed.hostname != "datafeed.dukascopy.com" or parsed.username or parsed.password
     or parsed.port not in (None,443) or parsed.query or parsed.fragment): raise ValueError("Dukascopy probe URL violates network policy")
 if not re.fullmatch(r"/datafeed/[A-Z0-9]{3,12}/20[0-9]{2}/(?:0[0-9]|1[01])/(?:0[1-9]|[12][0-9]|3[01])/(?:[01][0-9]|2[0-3])h_ticks\.bi5", parsed.path): raise ValueError("Dukascopy probe URL must be an official UTC BI5 hour path")
 return value
def cmd(*args:str)->str:
 return subprocess.run(args,cwd=REPO,capture_output=True,text=True,timeout=15).stdout.strip()
def fetch(url:str)->dict:
 req=urllib.request.Request(url,headers={"User-Agent":"simple-research-mcp/0.1 (research-only)"})
 with urllib.request.urlopen(req,timeout=TIMEOUT) as r: return json.loads(r.read().decode("utf-8"))
def result(value:object, error:bool=False)->dict:
 text=json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)
 return {"content":[{"type":"text","text":text}],"structuredContent":value,"isError":error}
def call(name:str,args:dict)->dict:
 try:
  schema=_tool_schema(name)
  if schema is None: return result({"error":"unknown tool"},True)
  error=_matches(schema,args)
  if error: return _invalid(error)
  if name=="research_seed_library":
   rows=json.loads(CORPUS.read_text(encoding="utf-8")); topic=args.get("topic","").lower(); return result([x for x in rows if not topic or topic in json.dumps(x).lower()])
  if name=="research_search_papers":
   query=args["query"].strip(); limit=min(max(int(args.get("limit",8)),1),20); data=fetch("https://api.crossref.org/works?rows=%d&query=%s"%(limit,urllib.parse.quote(query)))
   rows=[{"doi":x.get("DOI"),"title":(x.get("title")or[""])[0],"year":(x.get("published-print",x.get("published-online",{})).get("date-parts",[[None]])[0][0]),"venue":(x.get("container-title")or[""])[0]} for x in data["message"]["items"]]; return result(rows)
  if name=="research_doi_metadata":
   doi=args["doi"].strip().lower();
   if not re.fullmatch(r"10\.\S+/.+",doi): return result({"error":"doi must begin 10."},True)
   x=fetch("https://api.crossref.org/works/"+urllib.parse.quote(doi,safe=""))["message"]; return result({"doi":x.get("DOI"),"title":(x.get("title")or[""])[0],"publisher":x.get("publisher"),"type":x.get("type"),"url":x.get("URL")})
  if name=="research_evidence_check":
   claim=args["claim"].strip(); blocked=[]
   if re.search(r"profit|tradable|promot",claim,re.I): blocked.append("claim requires independent out-of-sample and execution evidence; paper alone is insufficient")
   return result({"claim":claim,"doi":args["doi"],"method":args["method"],"limitations":args["limitations"],"promotion_certified":False,"blockers":blocked,"next":"record source, causal timing, data scope, comparator, and holdout protocol"})
  if name=="agent_repo_snapshot":
   return result({"root":str(REPO),"branch":cmd("git","branch","--show-current"),"head":cmd("git","rev-parse","HEAD"),"status":cmd("git","status","--porcelain").splitlines()})
  if name=="agent_diff_scope":
   paths=cmd("git","diff","--cached" if args.get("staged") else "HEAD","--name-only").splitlines(); risky=[p for p in paths if re.search(r"(^data/|^artifacts/|secret|\.env|\.key$|\.pem$)",p,re.I)]
   return result({"staged":bool(args.get("staged")),"paths":paths,"risk_paths":risky,"safe_to_autocommit":not risky,"note":"Ownership requires human/agent assignment; this tool does not grant it."})
  if name=="agent_find_tests":
   query=args["query"]; tests=[]
   for p in (REPO/"tests").rglob("test_*.py"):
    if query.lower() in p.read_text(encoding="utf-8",errors="ignore").lower(): tests.append(str(p.relative_to(REPO)).replace("\\","/"))
   return result({"query":query,"tests":tests[:100]})
  if name=="agent_receipt_summary":
   receipts=[]
   for p in sorted((REPO/".agents"/"receipts").glob("*.json"),reverse=True)[:int(args.get("limit",20))]:
    try:
     item=json.loads(p.read_text(encoding="utf-8-sig")); claimed=item.pop("receipt_sha256",None)
     item["receipt_hash_verified"]=isinstance(claimed,str) and hashlib.sha256(json.dumps(item,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()==claimed
     log=REPO/".agents"/"receipts"/str(item.get("log_path", ""))
     item["log_hash_verified"]=log.is_file() and item.get("log_sha256")==hashlib.sha256(log.read_bytes()).hexdigest()
     receipts.append(item)
    except Exception: receipts.append({"file":p.name,"error":"invalid JSON"})
   return result(receipts)
  if name=="agent_tool_health":
   hooks=Path.home()/".codex"/"hooks.json"; hook_state="missing" if not hooks.exists() else ("empty" if json.loads(hooks.read_text()).get("hooks")=={} else "configured")
   return result({"python":sys.version.split()[0],"git":cmd("git","--version"),"node":cmd("node","--version"),"perplexity_key_present":bool(os.getenv("PERPLEXITY_API_KEY")),"firecrawl_key_present":bool(os.getenv("FIRECRAWL_API_KEY")),"codex_hooks":hook_state})
  if name=="quant_protocol_gate":
   missing=[key for key,value in args.items() if value is not True]
   return result({"promotion_certified":False,"protocol_ready":not missing,"missing_gates":missing,"next":"Resolve every missing gate; then run independent sealed evaluation."})
  if name=="physics_density_matrix_check":
   matrix=args["matrix"]; tol=float(args.get("tolerance",1e-9)); n=len(matrix)
   if not all(isinstance(row,list) and len(row)==n for row in matrix): return result({"error":"matrix must be square"},True)
   a=[[complex(value) for value in row] for row in matrix]; hermitian=max(abs(a[i][j]-a[j][i].conjugate()) for i in range(n) for j in range(n)); trace=sum(a[i][i] for i in range(n));
   return result({"dimension":n,"hermiticity_error":hermitian,"trace_real":trace.real,"trace_imag":trace.imag,"hermitian":hermitian<=tol,"trace_one":abs(trace-1)<=tol,"positivity":"not evaluated without a Hermitian eigensolver","physicality_certified":False})
  if name=="market_dukascopy_history_url":
   inst=args["instrument"].replace("/","").upper(); year=int(args["year"]); month=int(args["month"])-1; day=int(args["day"]); hour=int(args["hour_utc"])
   url=f"https://datafeed.dukascopy.com/datafeed/{inst}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"; return result({"url":url,"provider":"Dukascopy","timezone":"UTC","format":"BI5 compressed tick segment","month_zero_based":True,"downloaded":False})
  if name=="market_dukascopy_probe":
   url=_dukascopy_url(args["url"]); req=urllib.request.Request(url,headers={"Range":"bytes=0-0","User-Agent":"simple-market-mcp/0.1"}); opener=urllib.request.build_opener(_NoRedirect)
   with opener.open(req,timeout=min(TIMEOUT,15)) as response:
    body=response.read(1)
    return result({"url":url,"status":response.status,"content_length":response.headers.get("Content-Length"),"content_range":response.headers.get("Content-Range"),"content_type":response.headers.get("Content-Type"),"downloaded_payload_bytes":len(body)})
  if name=="market_tradingview_chart_url":
   symbol=args["symbol"].upper(); return result({"url":"https://www.tradingview.com/chart/?symbol="+urllib.parse.quote(symbol,safe=":"),"symbol":symbol,"data_api_called":False,"execution_warning":"Chart feed may differ from broker executable bid/ask feed."})
  if name=="market_feed_contract":
   blockers=[]
   if args["price_side"]=="unknown": blockers.append("price side unknown")
   if args["timezone"].upper() not in {"UTC","ETC/UTC"}: blockers.append("timezone not explicitly UTC")
   if not args["execution_feed_same"]: blockers.append("chart/data feed differs from execution feed")
   return result({"feed_contract_complete":not blockers,"blockers":blockers,"trading_certified":False})
  return result({"error":"unknown tool"},True)
 except (ValueError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError): return result({"error":"request rejected or unavailable","next":"check documented input requirements and retry"},True)
 except Exception: return result({"error":"tool failed","next":"check documented input requirements and retry"},True)
def main()->None:
 for line in sys.stdin:
  try:
   req=json.loads(line); method=req.get("method"); ident=req.get("id")
   if method=="initialize": out={"protocolVersion":req.get("params",{}).get("protocolVersion","2025-06-18"),"capabilities":{"tools":{}},"serverInfo":{"name":"simple-research","version":"0.1.0"}}
   elif method=="tools/list": out={"tools":TOOLS}
   elif method=="tools/call":
    out=call(req["params"]["name"],req["params"].get("arguments",{}))
    if out.pop("_invalid_params",False):
     print(json.dumps({"jsonrpc":"2.0","id":ident,"error":{"code":-32602,"message":out["content"][0]["text"]}}),flush=True); continue
   elif method=="notifications/initialized": continue
   else: print(json.dumps({"jsonrpc":"2.0","id":ident,"error":{"code":-32601,"message":"unknown method"}}),flush=True); continue
   print(json.dumps({"jsonrpc":"2.0","id":ident,"result":out},ensure_ascii=True,allow_nan=False),flush=True)
  except Exception as e: print(json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":str(e)}}),flush=True)
if __name__=="__main__": main()
