"""Compare full INT8 with FP32 on held-out SX images using the same CPU runtime."""
from pathlib import Path
import argparse, collections, json, statistics, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sx_int8_common import preprocess,truth,metrics
import cv2
import numpy as np
import openvino as ov


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--publish',type=Path,required=True)
    args=parser.parse_args();args.publish.mkdir(parents=True,exist_ok=True)
    core=ov.Core()
    config={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32','SCHEDULING_CORE_TYPE':'ANY_CORE','ENABLE_CPU_PINNING':True,'ENABLE_HYPER_THREADING':False,'PERF_COUNT':True}
    compiled={};requests={};runtime={}
    for key,path in [('FP32',ROOT/'models/1003_010_SX.onnx'),('INT8',args.model)]:
        m=core.read_model(path);m.reshape({m.input(0):[1,3,1024,1024]})
        compiled[key]=core.compile_model(m,'CPU',config);requests[key]=compiled[key].create_infer_request()
        runtime[key]=[{'name':n.get_friendly_name(),**{k:str(v.value) for k,v in n.get_rt_info().items()}} for n in compiled[key].get_runtime_model().get_ordered_ops()]
        (args.publish/(key+'_runtime.json')).write_text(json.dumps(runtime[key],indent=2),encoding='utf-8')
        print('COMPILED',key,dict(collections.Counter((n.get('layerType',''),n.get('runtimePrecision','')) for n in runtime[key] if n.get('layerType') in {'Subgraph','FullyConnected','Convolution','MatMul'})),flush=True)
    images=sorted(p for p in args.data.glob('*.jpg') if p.with_suffix('.json').exists())
    assert len(images)==3
    x=preprocess(images[0])[0]
    for key in requests:
        t=time.perf_counter();requests[key].infer({compiled[key].input(0):x});print('WARMED',key,time.perf_counter()-t,flush=True)
    report={'mode':'native OpenVINO CPU; three annotated held-out images','config':config,'runtime':ov.__version__,'input_shape':[1,3,1024,1024],'foreground_mapping':'sx=channel0 (black); mapping fixed from original FP32 4_8 before INT8 evaluation, IoU 0.944157 vs 0.007185 for channel1','requests':[],'notes':['Not an HTTP or C++ latency benchmark.','Profiling enabled for both models. Node counters are runtime-reported statistics, not exact per-request wall-clock times.','Static QDQ of every Conv and MatMul, including both attention products; non-GEMM operations remain floating point.','Only three annotated images; this is a preliminary precision check.']}
    def save(): (args.publish/'SX_INT8_native_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for index,path in enumerate(images):
        masks={};gt=truth(path) if path.with_suffix('.json').exists() else None
        for key in (['FP32','INT8'] if index%2==0 else ['INT8','FP32']):
            start=time.perf_counter();x,size,content=preprocess(path);pre_ms=(time.perf_counter()-start)*1000
            t=time.perf_counter();result=requests[key].infer({compiled[key].input(0):x});infer_ms=(time.perf_counter()-t)*1000
            seg_index=next(i for i,o in enumerate(compiled[key].outputs) if list(o.shape)==[1,2,1024,1024])
            logits=np.asarray(result[compiled[key].output(seg_index)])[0]
            assert np.all(np.isfinite(logits))
            t=time.perf_counter();left,top,width,height=content
            scores=[cv2.resize(logits[c,top:top+height,left:left+width],size,interpolation=cv2.INTER_LINEAR) for c in range(2)]
            mask=scores[1]>scores[0];post_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();colored=np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2);ok,png=cv2.imencode('.png',colored);assert ok
            encode_ms=(time.perf_counter()-t)*1000;total_ms=(time.perf_counter()-start)*1000
            stem=path.stem
            (args.publish/(stem+'_'+key+'.png')).write_bytes(png.tobytes())
            masks[key]=mask
            entry={'image':str(path),'name':stem,'backend':key,'preprocess_ms':pre_ms,'inference_ms':infer_ms,'postprocess_ms':post_ms,'encode_ms':encode_ms,'local_total_ms':total_ms,'white_percent':float(mask.mean()*100)}
            if gt is not None:entry['annotation_metrics']=metrics(~mask,gt)
            report['requests'].append(entry);save();print('RESULT',json.dumps(entry,ensure_ascii=False),flush=True)
        changed=masks['FP32']!=masks['INT8']
        report.setdefault('agreement',[]).append({'image':str(path),'changed_pixels':int(changed.sum()),'changed_percent':float(changed.mean()*100)})
        diff=np.zeros((*changed.shape,3),dtype=np.uint8);diff[masks['FP32']]=[255,255,255];diff[changed]=[0,0,255]
        cv2.imencode('.png',diff)[1].tofile(args.publish/(stem+'_difference.png'))
    for key,req in requests.items():
        rows=[{'name':p.node_name,'type':p.node_type,'exec_type':p.exec_type,'real_ms':p.real_time.total_seconds()*1000} for p in req.get_profiling_info()]
        (args.publish/(key+'_operator_profile.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report['summary']={key:{'median_inference_ms':statistics.median(e['inference_ms'] for e in report['requests'] if e['backend']==key),'median_local_total_ms':statistics.median(e['local_total_ms'] for e in report['requests'] if e['backend']==key),'mean_iou':statistics.mean(e['annotation_metrics']['iou'] for e in report['requests'] if e['backend']==key and 'annotation_metrics' in e),'mean_dice':statistics.mean(e['annotation_metrics']['dice'] for e in report['requests'] if e['backend']==key and 'annotation_metrics' in e)} for key in requests}
    report['status']='completed';save();print('SUMMARY',json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()

