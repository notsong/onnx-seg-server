"""Full Conv/MatMul static INT8 QDQ for SX; original ONNX remains unchanged."""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sx_int8_common import preprocess
import onnx
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibraterBase,MinMaxCalibrater
from onnxruntime.quantization import CalibrationDataReader,CalibrationMethod,QuantFormat,QuantType,quantize_static

class Images(CalibrationDataReader):
    def __init__(self,paths):self.paths=iter(paths);self.count=0
    def get_next(self):
        p=next(self.paths,None)
        if p is None:return None
        self.count+=1;print('CALIBRATE',self.count,p.name,flush=True)
        return {'input':preprocess(p)[0]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    images=sorted(p for p in args.data.glob('*.jpg'))
    calibration=[p for p in images if not p.with_suffix('.json').exists()]
    validation=[p for p in images if p.with_suffix('.json').exists()]
    original=ROOT/'models/1003_010_SX.onnx'
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    m=onnx.load(original)
    for v in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output):
        for d in v.type.tensor_type.shape.dim:
            if d.dim_param=='batch_size':d.dim_value=1
    # Keep MatMul/Conv unmerged so all attention products stay in the quantization scope.
    m=onnx.shape_inference.infer_shapes(m)
    fixed=args.output/'SX_FP32_batch1.onnx';onnx.save(m,fixed)
    original_ops=dict(collections.Counter(n.op_type for n in m.graph.node))
    target=args.output/'1003_010_SX_INT8_QDQ.onnx'
    started=time.perf_counter()
    print('QDQ_START',len(calibration),'calibration',len(validation),'held out',flush=True)
    # Calibration exposes attention matrices and disables graph fusion. Avoid a
    # large persistent ORT arena; the normal service does not use this setting.
    def calibration_session(calibrator):
        options=ort.SessionOptions()
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern=False
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        calibrator.infer_session=ort.InferenceSession(calibrator.augmented_model_path,sess_options=options,providers=calibrator.execution_providers)
    # Put scalar range reductions immediately after their producer. Appending
    # all of them at the end keeps 12 huge attention matrices alive together.
    original_augment=MinMaxCalibrater.augment_graph
    def memory_efficient_augment(calibrator):
        count=len(calibrator.model.graph.node)
        original_augment(calibrator)
        nodes=list(calibrator.model.graph.node)
        pending=nodes[count:];ordered=[]
        ready={x.name for x in calibrator.model.graph.input}|{x.name for x in calibrator.model.graph.initializer}
        def drain():
            nonlocal pending
            while True:
                available=[n for n in pending if all(not x or x in ready for x in n.input)]
                if not available:return
                for n in available:
                    ordered.append(n);ready.update(n.output);pending.remove(n)
        drain()
        for node in nodes[:count]:
            ordered.append(node);ready.update(node.output);drain()
        assert not pending
        del calibrator.model.graph.node[:]
        calibrator.model.graph.node.extend(ordered)
        onnx.save(calibrator.model,calibrator.augmented_model_path)
    MinMaxCalibrater.augment_graph=memory_efficient_augment
    CalibraterBase.create_inference_session=calibration_session
    quantize_static(str(fixed),str(target),Images(calibration),quant_format=QuantFormat.QDQ,
        op_types_to_quantize=['Conv','MatMul'],per_channel=True,reduce_range=True,
        activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=['CPUExecutionProvider'],extra_options={'MatMulConstBOnly':False,'WeightSymmetric':True,'ActivationSymmetric':False,'CalibMaxIntermediateOutputs':None})
    q=onnx.load(target)
    producers={o:n for n in q.graph.node for o in n.output}
    coverage=collections.defaultdict(list)
    for n in q.graph.node:
        if n.op_type in ['Conv','MatMul']:
            coverage[n.op_type].append({'name':n.name,'inputs_0_1_dequantized':[producers.get(x).op_type if x in producers else 'initializer/input' for x in n.input[:2]]})
    assert all(n['inputs_0_1_dequantized']==['DequantizeLinear','DequantizeLinear'] for group in coverage.values() for n in group)
    assert len(coverage['MatMul'])==original_ops['MatMul'] and len(coverage['Conv'])==original_ops['Conv']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_hash
    report={'kind':'full static activation/weight INT8 QDQ; Conv and every MatMul including QK and attention*V','runtime_quantization':'onnxruntime 1.22.0','calibration':[str(p) for p in calibration],'validation':[str(p) for p in validation],'model_sha256':original_hash,'model':str(target),'quantization_seconds':time.perf_counter()-started,'model_bytes':target.stat().st_size,'original_op_counts':original_ops,'quantized_op_counts':dict(collections.Counter(n.op_type for n in q.graph.node)),'coverage':coverage,'dtype':'uint8 activation, int8 per-channel constant weights, reduce_range enabled for AVX2 overflow mitigation','notes':['All original Conv and MatMul inputs are surrounded by QDQ. Actual integer kernels still require runtime inspection.','Input/output are float32; normalization, Softmax, GELU and other non-Conv/MatMul operations remain floating point.','A smaller model file is not proof of faster inference.']}
    (args.output/'quantization_QDQ_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('QDQ_DONE',report['quantization_seconds'],report['model_bytes'],{k:len(v) for k,v in coverage.items()},flush=True)
if __name__=='__main__':main()



