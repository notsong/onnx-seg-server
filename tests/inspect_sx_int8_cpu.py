"""Verify actual ONNX Runtime CPU integer operators on one held-out SX image."""
from pathlib import Path
import argparse,collections,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sx_int8_common import preprocess
import onnx,onnxruntime as ort

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--publish',type=Path,required=True)
    args=p.parse_args();args.publish.mkdir(parents=True,exist_ok=True)
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    options.optimized_model_filepath=str(args.publish/'SX_INT8_ORT_optimized.onnx')
    options.enable_profiling=True
    options.profile_file_prefix=str(args.publish/'ORT_INT8_profile')
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    optimized=onnx.load(options.optimized_model_filepath)
    counts=dict(collections.Counter(n.op_type for n in optimized.graph.node))
    print('CPU_OPTIMIZED_OPS',json.dumps(counts),flush=True)
    x=preprocess(args.image)[0]
    timings=[]
    for i in range(2):
        t=time.perf_counter();session.run(None,{'input':x});timings.append(time.perf_counter()-t)
        print('CPU_PROFILE_RUN',i+1,timings[-1],flush=True)
    profile=Path(session.end_profiling())
    events=json.loads(profile.read_text(encoding='utf-8'))
    kernels=[e for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time')]
    actual=dict(collections.Counter((e.get('args',{}).get('op_name','')+'@'+e.get('args',{}).get('provider','')) for e in kernels))
    report={'runtime':ort.__version__,'model':str(args.model),'optimized_op_counts':counts,
            'executed_kernel_counts_two_runs':actual,'profile':str(profile),
            'profiling_run_seconds':timings,
            'note':'This verifies CPU kernel execution. Profiled native calls are not HTTP timings.'}
    assert counts.get('QLinearMatMul',0)==96,counts
    assert not counts.get('MatMul',0) and not counts.get('Conv',0),counts
    assert actual.get('QLinearMatMul@CPUExecutionProvider',0)==192,actual
    report['all_original_matmuls_integer_executed']=True
    (args.publish/'ORT_INT8_kernel_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CPU_INTEGER_KERNELS_CONFIRMED',json.dumps(actual),flush=True)
if __name__=='__main__':main()
