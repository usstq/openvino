# Model Creation in Runtime {#openvino_2_0_model_creation}

OpenVINO™ Runtime API 2.0 includes the nGraph engine as a common part. The `ngraph` namespace has been changed to `ov` but all other parts of the ngraph API have been preserved.
The code snippets below show how to change application code for migration to OpenVINO™ Runtime API 2.0.

### nGraph API

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ngraph.cpp ngraph:graph
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ngraph.py ngraph:graph
@endsphinxtab

@endsphinxtabset

### OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_graph.cpp ov:graph
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_graph.py ov:graph
@endsphinxtab

@endsphinxtabset

**See also:**
- [Hello Model Creation C++ Sample](../../../samples/cpp/model_creation_sample/README.md)
- [Hello Model Creation Python Sample](../../../samples/python/model_creation_sample/README.md)
