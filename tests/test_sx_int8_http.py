"""Sequential, isolated C++ HTTP validation; avoids two large CPU sessions at once."""
from pathlib import Path
import argparse,io,json,os,re,statistics,sys,time,urllib.request,uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from sx_int8_common import truth,metrics
from self_test import Server,multipart
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--publish',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--threads',type=int,default=0)
    args=p.parse_args();args.publish.mkdir(parents=True,exist_ok=True)
    run=ROOT/'tests/generated'/('sx-int8-isolated-'+uuid.uuid4().hex[:8])
    cfg=json.loads((ROOT/'config/1003_010.json').read_text(encoding='utf-8-sig'))
    paths=sorted(p for p in args.data.glob('*.jpg') if p.with_suffix('.json').exists())
    assert len(paths)==3
    report={'mode':'C++ HTTP; one warmed CPU service resident at a time; FP32 block then INT8 block',
        'backend':'onnxruntime CPU','executable':str(args.exe),'threads':args.threads,
        'cpu':'Intel Core 7 150U','input_shape':[1,3,1024,1024],
        'foreground_mapping':'sx=channel0 (black), fixed before INT8 evaluation',
        'config':cfg,'run_directory':str(run),'requests':[],
        'notes':['Startup/warmup excluded. Local HTTP upload/PNG download included.',
                 'Sequential blocks avoid simultaneous large CPU sessions; order may affect thermal/load conditions.',
                 'Only three held-out annotated images; preliminary accuracy evaluation.']}
    masks={}
    def save(): (args.publish/'SX_INT8_HTTP_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for key,model in [('FP32',ROOT/'models/1003_010_SX.onnx'),('INT8',args.model)]:
        root=run/key;(root/'config').mkdir(parents=True);(root/'models').mkdir()
        local=json.loads(json.dumps(cfg));local['model_detail']['model_name']=model.name
        (root/'config/1003_010.json').write_text(json.dumps(local),encoding='utf-8')
        os.link(model,root/'models'/model.name)
        print('START',key,flush=True)
        server=Server(args.exe,root,threads=args.threads,backend='onnxruntime',startup_timeout=600,request_timeout=600)
        try:
            print('READY',key,flush=True)
            for path in paths:
                body,kind=multipart([('catagory','1003'),('item','10')],[('file',path.read_bytes())])
                t=time.perf_counter()
                req=urllib.request.Request(server.url+'/infer',data=body,headers={'Content-Type':kind})
                with urllib.request.urlopen(req,timeout=600) as response:status,headers,payload=response.status,response.headers,response.read()
                client=(time.perf_counter()-t)*1000
                assert status==200 and headers['Content-Type']=='image/png'
                assert headers['X-Model-Key']=='1003_010' and int(headers['X-Tile-Count'])==1
                gt=truth(path)
                with Image.open(io.BytesIO(payload)) as image:mask=np.asarray(image.convert('RGB'))
                assert mask.shape==(*gt.shape,3)
                assert np.all(np.all(mask==0,axis=2)|np.all(mask==255,axis=2))
                pred=mask[:,:,0]==0;masks[(key,path.name)]=pred
                out=args.publish/(path.stem+'_'+key+'_HTTP.png');out.write_bytes(payload)
                row={'image':path.name,'model':key,'inference_ms':float(headers['X-Inference-Ms']),
                     'server_total_ms':float(headers['X-Total-Ms']),'client_ms':client,
                     'mask':str(out),'annotation_metrics':metrics(pred,gt),'stage_ms':{}}
                log=server.log_path.read_text(encoding='utf-8',errors='replace')
                for stage in ['decode','preprocess','inference','postprocess','encode','total']:
                    values=re.findall(stage+r'_ms=([0-9.]+)',log)
                    if values:row['stage_ms'][stage]=float(values[-1])
                report['requests'].append(row);save();print('RESULT',json.dumps(row,ensure_ascii=False),flush=True)
        finally:server.close()
    report['summary']={key:{
        'median_inference_ms':statistics.median(r['inference_ms'] for r in report['requests'] if r['model']==key),
        'median_client_ms':statistics.median(r['client_ms'] for r in report['requests'] if r['model']==key),
        'client_range_ms':[min(r['client_ms'] for r in report['requests'] if r['model']==key),max(r['client_ms'] for r in report['requests'] if r['model']==key)],
        'mean_iou':statistics.mean(r['annotation_metrics']['iou'] for r in report['requests'] if r['model']==key),
        'mean_dice':statistics.mean(r['annotation_metrics']['dice'] for r in report['requests'] if r['model']==key)}
        for key in ['FP32','INT8']}
    report['agreement']=[{'image':p.name,'changed_percent':float(np.mean(masks[('FP32',p.name)]!=masks[('INT8',p.name)])*100)} for p in paths]
    report['all_INT8_under_5_seconds']=all(r['client_ms']<=5000 for r in report['requests'] if r['model']=='INT8')
    report['status']='completed';save();print('SUMMARY',json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
