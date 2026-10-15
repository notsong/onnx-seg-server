"""Near-adjacent CPU/OpenVINO HTTP timing comparison; same pre/postprocessing."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from self_test import Server, multipart
import argparse, hashlib, io, json, os, re, shutil, statistics, time, urllib.request, uuid
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--openvino-exe",type=Path,required=True)
    p.add_argument("--publish",type=Path,required=True)
    args=p.parse_args()
    args.publish.mkdir(parents=True,exist_ok=True)
    run=ROOT/"tests"/"generated"/("sx-openvino-paired-"+uuid.uuid4().hex[:8])
    (run/"config").mkdir(parents=True)
    (run/"models").mkdir()
    cfg=ROOT/"config"/"1003_010.json"
    model=ROOT/"models"/"1003_010_SX.onnx"
    shutil.copy2(cfg,run/"config"/cfg.name)
    os.link(model,run/"models"/model.name)
    def sha(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f,"sha256").hexdigest()
    body,kind=multipart([("catagory","1003"),("item","10")],[("file",args.image.read_bytes())])
    body=body.replace(b'filename="image.png"\r\nContent-Type: image/png',
                      b'filename="image.jpg"\r\nContent-Type: image/jpeg',1)
    with Image.open(args.image) as image:
        size=image.size
    groups=[]
    baseline=None
    report={"tested_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "mode":"three interleaved pairs after both services warmed",
            "cpu":"Intel Core 7 150U","threads":0,"precision":"FP32",
            "run_directory":str(run),"model_sha256":sha(model),"image_sha256":sha(args.image),
            "image_size":list(size),"config":json.loads(cfg.read_text(encoding="utf-8")),
            "openvino_settings":{"batch_size_override":1,"device_type":"CPU","precision":"FP32",
                "num_streams":1,"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f32",
                "SCHEDULING_CORE_TYPE":"ANY_CORE","ENABLE_CPU_PINNING":True,"ENABLE_HYPER_THREADING":False},
            "groups":groups,"pairs":[],
            "notes":["Two warmed C++ services remain loaded; only one request runs at a time.",
                     "Ordinary CPU uses original root executable/CPU DLL; OpenVINO uses staged native DLLs.",
                     "Both paths use identical C++ image processing and original SX ONNX weights.",
                     "Pixel agreement with CPU is not ground-truth accuracy. Background load can still vary."]}
    def save():
        text=json.dumps(report,indent=2,ensure_ascii=False)+"\n"
        (run/"report.json").write_text(text,encoding="utf-8")
        (args.publish/"SX_openvino_paired_report.json").write_text(text,encoding="utf-8")
    servers={}
    try:
        for name,exe,backend in [("cpu",ROOT/"onnx-seg-server.exe",None),
                                ("openvino",args.openvino_exe,"openvino")]:
            print("START "+name,flush=True)
            started=time.perf_counter()
            server=Server(exe,run,threads=0,backend=backend,startup_timeout=600,request_timeout=600)
            servers[name]=server
            group={"name":name,"executable":str(exe),"executable_sha256":sha(exe),
                   "startup_and_warmup_seconds":round(time.perf_counter()-started,3),
                   "log":str(server.log_path),"requests":[]}
            groups.append(group)
            print("READY "+name+" "+str(group["startup_and_warmup_seconds"]),flush=True)
        save()
        for index in range(3):
            pair={}
            # Reverse the middle pair to reduce systematic ordering bias.
            order=groups if index!=1 else list(reversed(groups))
            for group in order:
                name=group["name"]
                server=servers[name]
                started=time.perf_counter()
                req=urllib.request.Request(server.url+"/infer",data=body,headers={"Content-Type":kind})
                with urllib.request.urlopen(req,timeout=600) as response:
                    status,headers,payload=response.status,response.headers,response.read()
                client=(time.perf_counter()-started)*1000
                assert status==200 and headers["Content-Type"]=="image/png"
                assert headers["X-Model-Key"]=="1003_010" and int(headers["X-Tile-Count"])==1
                with Image.open(io.BytesIO(payload)) as image:
                    assert image.size==size
                    mask=np.array(image.convert("RGB"))
                assert np.all(np.all(mask==0,axis=2)|np.all(mask==255,axis=2))
                if baseline is None:
                    baseline=mask.copy()
                changed=int(np.count_nonzero(np.any(mask!=baseline,axis=2)))
                entry={"pair":index+1,"inference_ms":float(headers["X-Inference-Ms"]),
                       "server_total_ms":float(headers["X-Total-Ms"]),"client_ms":round(client,3),
                       "changed_pixels_vs_cpu":changed,"changed_percent_vs_cpu":round(changed/(size[0]*size[1])*100,6),
                       "mask_sha256":hashlib.sha256(payload).hexdigest(),"stage_ms":{}}
                log=server.log_path.read_text(encoding="utf-8",errors="replace")
                for part in ("decode","preprocess","inference","postprocess","encode","total"):
                    values=re.findall(part+r"_ms=([0-9.]+)",log)
                    if values:
                        entry["stage_ms"][part]=float(values[-1])
                if not group["requests"]:
                    mask_path=args.publish/("SX_mask_"+name+".png")
                    mask_path.write_bytes(payload)
                    group["mask"]=str(mask_path)
                group["requests"].append(entry)
                pair[name]=entry["inference_ms"]
                save()
                print("RESULT "+json.dumps({"group":name,**entry}),flush=True)
            pair["speed_ratio_cpu_over_openvino"]=pair["cpu"]/pair["openvino"]
            report["pairs"].append(pair)
            save()
        for group in groups:
            values=[r["inference_ms"] for r in group["requests"]]
            group["median_inference_ms"]=statistics.median(values)
            group["median_server_total_ms"]=statistics.median(r["server_total_ms"] for r in group["requests"])
            group["range_inference_ms"]=[min(values),max(values)]
        report["status"]="passed"
        report["median_paired_speed_ratio"]=statistics.median(p["speed_ratio_cpu_over_openvino"] for p in report["pairs"])
        assert report["model_sha256"]==sha(model)
        save()
        print("SUMMARY "+json.dumps({g["name"]:{"median_inference_ms":g["median_inference_ms"],
            "median_server_total_ms":g["median_server_total_ms"],"range_inference_ms":g["range_inference_ms"]}
            for g in groups}),flush=True)
        print("PAIRED_SPEED_RATIO "+str(report["median_paired_speed_ratio"]),flush=True)
    finally:
        for server in servers.values():
            server.close()
if __name__=="__main__":
    main()
