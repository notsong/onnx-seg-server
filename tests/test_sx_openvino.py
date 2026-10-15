"""Compare the real C++ HTTP SX pipeline across CPU and OpenVINO CPU FP32."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from self_test import Server, multipart
import argparse, collections, hashlib, io, json, os, re, shutil, statistics, time, urllib.request, uuid
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--openvino-exe",type=Path,required=True)
    parser.add_argument("--publish",type=Path,required=True)
    args=parser.parse_args()
    args.publish.mkdir(parents=True,exist_ok=True)
    model=ROOT/"models"/"1003_010_SX.onnx"
    cfg=ROOT/"config"/"1003_010.json"
    run=ROOT/"tests"/"generated"/("sx-openvino-"+uuid.uuid4().hex[:8])
    (run/"models").mkdir(parents=True)
    (run/"config").mkdir()
    os.link(model,run/"models"/model.name)
    shutil.copy2(cfg,run/"config"/cfg.name)
    shutil.copy2(args.image,run/"SX_input.jpg")
    with Image.open(args.image) as image:
        size=image.size
    body,kind=multipart([("catagory","1003"),("item","10")],[("file",args.image.read_bytes())])
    body=body.replace(b'filename="image.png"\r\nContent-Type: image/png',
                      b'filename="image.jpg"\r\nContent-Type: image/jpeg',1)
    baseline=None
    report={"tested_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "cpu":"Intel Core 7 150U","threads":0,"precision":"FP32",
            "model":str(model),"model_sha256":sha(model),"image":str(args.image),
            "image_sha256":sha(args.image),"image_size":list(size),
            "config":json.loads(cfg.read_text(encoding="utf-8")),
            "run_directory":str(run),"groups":[],
            "notes":["Real C++ HTTP requests; identical image decoding/preprocess/score restoration/PNG flow.",
                     "Each server loads and warms before three measured sequential requests.",
                     "Ordinary CPU uses ORT_ENABLE_ALL; OpenVINO uses ORT_DISABLE_ALL and one CPU FP32 LATENCY stream.",
                     "A final ordinary-CPU request checks load drift; no other processes or power settings are changed.",
                     "Mask difference and IoU versus ordinary CPU quantify numerical agreement, not ground-truth accuracy."]}
    def save():
        text=json.dumps(report,indent=2,ensure_ascii=False)+"\n"
        (run/"report.json").write_text(text,encoding="utf-8")
        (args.publish/"SX_openvino_comparison_report.json").write_text(text,encoding="utf-8")
    def group(name,exe,backend,repeats):
        nonlocal baseline
        print("START "+name,flush=True)
        started=time.perf_counter()
        server=Server(exe,run,threads=0,backend=backend,startup_timeout=600,request_timeout=600)
        data={"name":name,"backend":backend or "onnxruntime","exe":str(exe),"exe_sha256":sha(exe),
              "startup_and_warmup_seconds":round(time.perf_counter()-started,3),
              "log":str(server.log_path),"requests":[]}
        print("READY "+name+" startup_s="+str(data["startup_and_warmup_seconds"]),flush=True)
        try:
            for index in range(repeats):
                started=time.perf_counter()
                request=urllib.request.Request(server.url+"/infer",data=body,headers={"Content-Type":kind})
                with urllib.request.urlopen(request,timeout=600) as response:
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
                per_class={}
                for c in (0,1):
                    expected=baseline[:,:,0]==255*c
                    actual=mask[:,:,0]==255*c
                    union=int(np.count_nonzero(expected|actual))
                    per_class[str(c)]={"pixels":int(actual.sum()),
                        "iou_vs_cpu":float(np.count_nonzero(expected&actual)/union) if union else 1.0}
                entry={"index":index+1,"http_status":status,
                       "inference_ms":float(headers["X-Inference-Ms"]),
                       "server_total_ms":float(headers["X-Total-Ms"]),
                       "client_ms":round(client,3),"changed_pixels_vs_cpu":changed,
                       "changed_percent_vs_cpu":round(changed/(size[0]*size[1])*100,6),
                       "classes":per_class,"mask_sha256":hashlib.sha256(payload).hexdigest(),"stage_ms":{}}
                log=server.log_path.read_text(encoding="utf-8",errors="replace")
                for part in ("decode","preprocess","inference","postprocess","encode","total"):
                    hits=re.findall(part+r"_ms=([0-9.]+)",log)
                    if hits:
                        entry["stage_ms"][part]=float(hits[-1])
                if index==0:
                    file=args.publish/("SX_mask_"+name+".png")
                    file.write_bytes(payload)
                    shutil.copy2(file,run/file.name)
                    data["mask"]=str(file)
                data["requests"].append(entry)
                print("RESULT "+json.dumps({"group":name,**entry}),flush=True)
        finally:
            server.close()
        data["median_inference_ms"]=statistics.median(r["inference_ms"] for r in data["requests"])
        data["median_server_total_ms"]=statistics.median(r["server_total_ms"] for r in data["requests"])
        data["range_inference_ms"]=[min(r["inference_ms"] for r in data["requests"]),
                                    max(r["inference_ms"] for r in data["requests"])]
        report["groups"].append(data)
        save()
    save()
    group("cpu",ROOT/"onnx-seg-server.exe",None,3)
    group("openvino",args.openvino_exe,"openvino",3)
    group("cpu_recheck",ROOT/"onnx-seg-server.exe",None,1)
    report["status"]="passed"
    cpu,ov=report["groups"][:2]
    report["speed_ratio_cpu_over_openvino"]=cpu["median_inference_ms"]/ov["median_inference_ms"]
    report["openvino_time_reduction_percent"]=(1-ov["median_inference_ms"]/cpu["median_inference_ms"])*100
    assert sha(model)==report["model_sha256"]
    save()
    print("DONE "+str(args.publish/"SX_openvino_comparison_report.json"),flush=True)
    print("SUMMARY "+json.dumps({g["name"]:{
        "median_inference_ms":g["median_inference_ms"],"median_total_ms":g["median_server_total_ms"],
        "range_ms":g["range_inference_ms"],
        "changed_pixels_vs_cpu":g["requests"][0]["changed_pixels_vs_cpu"]
    } for g in report["groups"]}),flush=True)

if __name__=="__main__":
    main()
